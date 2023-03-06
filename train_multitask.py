from __future__ import print_function

import os
import random
import argparse
import torch.nn.parallel
import torch.optim as optim

from utils import Logger, create_dir, backup_code
from cifar.trainer import testing_loop, training_loop_multitask, inference

import cifar.models as models
from multitask_model import trace_model, get_basis_channels_from_t
import incremental_dataloader as data
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets

parser.add_argument('--jobid', type=str, default='test')
parser.add_argument('--arch', default='resnet18')
parser.add_argument('--add-bn-prev', type=str2bool, nargs='?', default=False)
parser.add_argument('--add-bn-next', type=str2bool, nargs='?', default=False)

parser.add_argument('-d', '--dataset', default='cifar100', type=str)
parser.add_argument('--data-path', default='../../data/CIFAR', type=str)
parser.add_argument('--increments', type=int, nargs='+', default=[10]*10)
parser.add_argument('--validation', default=0, type=int)

parser.add_argument('--random-classes', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--overflow', type=str2bool, nargs='?', const=True, default=False)

# parser.add_argument('--starting-tid', type=int, default=0)
# # checkpoint/132937_resnet32_multitask/model0_best.pth
# parser.add_argument('--pretrained-cp', type=str, default='')
parser.add_argument('-j', '--workers', default=0, type=int)
parser.add_argument('--compression', default=1.0, type=float)
# Task1 options
parser.add_argument('--display-gap', default=50, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150, 200], help='Decrease learning rate at these epochs.')
parser.add_argument('--lr', default=0.1, type=float)

parser.add_argument('--ft-epochs', default=1, type=int)
parser.add_argument('--ft-schedule', type=int, nargs='+', default=[100, 150, 200])
parser.add_argument('--ft-lr', default=0.01, type=float)
parser.add_argument('--ft-weight-decay', default=5e-4, type=float)

parser.add_argument('--train-batch', default=64, type=int)
parser.add_argument('--test-batch', default=100, type=int)
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--logs', default='logs', type=str, metavar='PATH', help='path to save the training logs (default: logs)')
parser.add_argument('--manual-seed', type=int, default=None, help='manual seed')

args = parser.parse_args()

# Random seed
if args.manual_seed is None:
     args.manual_seed = random.randint(1, 10000)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

def get_data_loaders(args):
    train_loaders = []
    test_loaders = []
    inc_dataset = data.IncrementalDataset(
        dataset_name=args.dataset,
        args=args,
        random_order=args.random_classes,
        shuffle=True,
        seed=args.manual_seed,
        batch_size=args.train_batch,
        workers=args.workers,
        validation_split=args.validation,
        increment=args.increments[0],
    )
    task_data = []
    for i in range(len(args.increments)):
        task_info, trl, vll, tsl = inc_dataset.new_task()

        train_loaders.append(trl)
        test_loaders.append(tsl)
    # args.increments = inc_dataset.increments

    return train_loaders, test_loaders

def train_task1(model, train_loaders, test_loaders, args, save_best):
    ###########################################################################
    ####################### Train Conv Model on Task 1 ########################
    ###########################################################################


    logger = Logger(dir_path=os.path.join(args.logs, args.jobid + '_' + args.arch), fname='task0',
                    keys=['time', 'acc1', 'acc5', 'ce_loss'])
    logger.one_time({'seed': args.manual_seed, 'comments': 'Train task 0'})
    logger.set_names(['lr', 'train_stats', 'test_stats'])
    print('\n\n'+ '_'*90 +'\n')
    print('Training task: 0')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    model = training_loop_multitask(model=model, optimizer=optimizer, task_id=0, train_loaders=train_loaders,
                                    test_loaders=test_loaders, logger=logger, args=args,
                                    save_best=save_best)

    num_conv, num_linear, in_channels, out_channels, basis_channels, layer_type = trace_model(model)
    _, _, basis_channels = get_basis_channels_from_t(model, [args.compression] * num_conv)
    print(basis_channels)

    # Create a multitask model with the basis channels estimated above
    mt_model = models.__dict__[args.arch + '_multitask'](basis_channels_list=basis_channels,
        add_bn_prev_list=args.add_bn_prev, add_bn_next_list=args.add_bn_next, num_classes=args.increments[0])
    # Initilize the task 1 parameters of multitask model using the weights of conv2d model
    print(mt_model)
    mt_model.cuda()
    mt_model.load_t1_weights(model)

    ###########################################################################
    ##################### Finetune Conv Model on Task 1 #######################
    ###########################################################################

    if args.compression < 1.0 or args.add_bn_prev: # Finetune

        logger = Logger(dir_path=os.path.join(args.logs, args.jobid + '_' + args.arch), fname='task0_ft',
                        keys=['time', 'acc1', 'acc5', 'ce_loss'])
        logger.one_time({'seed': args.manual_seed, 'comments': 'Finetune task 0'})
        logger.set_names(['lr', 'train_stats', 'test_stats'])

        print('\n\n###########################################\n')
        print('Finetuning task 0')
        [epochs, schedule, lr, weight_decay] = args.epochs, args.schedule, args.lr, args.weight_decay
        args.epochs, args.schedule, args.lr, args.weight_decay = args.ft_epochs, args.ft_schedule, args.ft_lr, args.ft_weight_decay

        optimizer = optim.SGD(mt_model.get_task_parameter(0), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        mt_model = training_loop_multitask(model=mt_model, optimizer=optimizer, task_id=0, train_loaders=train_loaders,
                                        test_loaders=test_loaders, logger=logger, args=args,
                                        save_best=save_best)

        [args.epochs, args.schedule, args.lr, args.weight_decay] = epochs, schedule, lr, weight_decay

    return mt_model

def main():
    print(args)
    args.num_class = sum(args.increments)

    args.checkpoint = os.path.join(args.checkpoint, args.jobid + '_' + args.arch)
    # backup_code(os.path.join('logs', args.jobid + '_' + args.arch, 'backup'), ['train_multitask.py', 'trainer.py', 'train_multitask.slurm'])
    create_dir([args.checkpoint, args.logs])

    model = models.__dict__[args.arch](num_classes=args.increments[0])
    model.set_task_id = lambda x: print('')
    model.cuda()

    train_loaders, test_loaders = get_data_loaders(args)

    class_incrimental_accuracy = []
    task_prediction_accuracy = []

    mt_model = train_task1(model=model, train_loaders=train_loaders, test_loaders=test_loaders, args=args, save_best=True)

    tmp, tmp1 = inference(test_loaders[:1], mt_model, args, 1)
    class_incrimental_accuracy.append(tmp)
    task_prediction_accuracy.append(tmp1)

    ###################################################################################################################
    ######################################### Train remaining tasks ###################################################
    ###################################################################################################################

    for i in range(1, len(args.increments)):
        # Setup logger
        logger = Logger(dir_path=os.path.join(args.logs, args.jobid + '_' + args.arch), fname='task'+str(i),
                        keys=['time', 'acc1', 'acc5', 'ce_loss'])
        logger.one_time({'seed': args.manual_seed, 'comments': 'Train task '+str(i)})
        logger.set_names(['lr', 'train_stats', 'test_stats'])
        print('\n\n'+ '_'*90 +'\n')
        print('Training task: ', i)

        # Add new task parameters to the model
        mt_model.add_task(copy_from=0, add_bn_prev_list=args.add_bn_prev, add_bn_next_list=args.add_bn_next, num_classes=args.increments[i])
        mt_model.set_task_id(i)
        mt_model.cuda()
        # print(mt_model)

        # Training model
        optimizer = optim.SGD(mt_model.get_task_parameter(i), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        mt_model = training_loop_multitask(model=mt_model, optimizer=optimizer, task_id=i, train_loaders=train_loaders, test_loaders=test_loaders, logger=logger, args=args,
                                        save_best=True)

        tmp, tmp1 = inference(test_loaders[:i+1], mt_model, args, i+1)
        class_incrimental_accuracy.append(tmp)
        task_prediction_accuracy.append(tmp1)

        print('Class Incrimental Accuracy : ', class_incrimental_accuracy)
        print('Avg. Class Incrimental Accuracy : ', sum(class_incrimental_accuracy) / len(class_incrimental_accuracy))
        print('Task Prediction Accuracy : ', task_prediction_accuracy)

    print('\n')
    print(args)

if __name__ == '__main__':
    main()