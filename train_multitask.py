from __future__ import print_function

import os
import random
import argparse
import torch.nn.parallel
import torch.optim as optim
import torch.nn as nn

from utils import Logger, create_dir, backup_code
from cifar.trainer import test, training_loop_multitask, inference

import cifar.models as models
from multitask_model import trace_model, get_basis_channels_from_t, display_stats
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

parser = argparse.ArgumentParser(description='Compressed Continual Learning')
# Datasets

parser.add_argument('--jobid', type=str, default='test')

parser.add_argument('--arch', default='resnet18')
parser.add_argument('--increments', type=int, nargs='+', default=[10]*10)
parser.add_argument('--add-bn-prev', type=str2bool, nargs='?', default=False)
parser.add_argument('--add-bn-next', type=str2bool, nargs='?', default=True)
parser.add_argument('--carry-all', type=str2bool, nargs='?', default=False)

parser.add_argument('--compression', default=0.95, type=float)
parser.add_argument('--growth-rate', default=0.1, type=float)

parser.add_argument('--resume', type=str2bool, nargs='?', default=False)
parser.add_argument('--starting-tid', type=str, default='0_ft', help='<tid> or <tid_ft>')
parser.add_argument('--pretrained-cps', type=str, default='./checkpoint/226429_resnet18/model0_best.pth,./checkpoint/227651_resnet18/model_0_ft_best.pth')

parser.add_argument('-d', '--dataset', default='cifar100', type=str)
parser.add_argument('--data-path', default='../../data/CIFAR', type=str)
parser.add_argument('-j', '--workers', default=0, type=int)
parser.add_argument('--train-batch', default=64, type=int)
parser.add_argument('--test-batch', default=100, type=int)

parser.add_argument('--validation', default=0, type=int)
parser.add_argument('--random-classes', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--overflow', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--class-order', type=int, nargs='+', default=[68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92,
        55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72,
        60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88,
        27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33])

parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150, 200], help='Decrease learning rate at these epochs.')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)

parser.add_argument('--ft-epochs', default=1, type=int)
parser.add_argument('--ft-schedule', type=int, nargs='+', default=[100, 150, 200])
parser.add_argument('--ft-lr', default=0.01, type=float)
parser.add_argument('--ft-weight-decay', default=5e-4, type=float)

# Checkpoints
parser.add_argument('--display-gap', default=3, type=int)
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

def get_logger(fname, comment):
    logger = Logger(dir_path=os.path.join(args.logs, args.jobid + '_' + args.arch), fname=fname,
                    keys=['time', 'acc1', 'acc5', 'ce_loss'])
    logger.one_time({'seed': args.manual_seed, 'comments': comment})
    logger.set_names(['lr', 'train_stats', 'test_stats'])

    return logger
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
        class_order = args.class_order
    )
    task_data = []
    for i in range(len(args.increments)):
        task_info, trl, vll, tsl = inc_dataset.new_task()

        train_loaders.append(trl)
        test_loaders.append(tsl)
    # args.increments = inc_dataset.increments

    return train_loaders, test_loaders

def empty_fun(x):
    pass

task_order = {}
for i in range(100):
    task_order[str(i)] = i * 2
    task_order[str(i) + '_ft'] = i * 2 + 1
def exec_block(args, tid):

    if task_order[args.starting_tid] <= task_order[tid]:
        return True
    else:
        return False

def resume(args, tid, model):
    file = args.pretrained_cps.split(',')[task_order[tid]]
    sd = torch.load(file)
    model.load_state_dict(sd)

    return model

def main():
    print(torch.__version__)
    print(torch.version.cuda)
    print(args)

    assert not (args.random_classes and args.resume), 'Cannot resume with random classes'
    assert sum(args.increments) == len(args.class_order), 'Number of classes and task split mismatch'

    args.checkpoint = os.path.join(args.checkpoint, args.jobid + '_' + args.arch)
    # backup_code(os.path.join('logs', args.jobid + '_' + args.arch, 'backup'), ['train_multitask.py', 'trainer.py', 'train_multitask.slurm'])
    create_dir([args.checkpoint, args.logs])
    args.num_class = sum(args.increments)

    train_loaders, test_loaders = get_data_loaders(args)

    ###########################################################################
    ####################### Train Conv Model on Task 1 ########################
    ###########################################################################

    model = models.__dict__[args.arch](num_classes=args.increments[0])
    model.set_task_id = empty_fun
    model.cuda()

    if exec_block(args, '0'):

        logger = get_logger(fname='task0', comment='Train task 0')
        print('\n\n' + '_' * 90 + '\n')
        print('Training task: 0')

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        model = training_loop_multitask(model=model, optimizer=optimizer, task_id=0, train_loaders=train_loaders,
                                        test_loaders=test_loaders, logger=logger, args=args,
                                        save_best=True)

    else:
        print('Resuming task: 0')
        model = resume(args, '0', model)

    num_conv, num_linear, in_channels, out_channels, basis_channels, layer_type = trace_model(model)
    _, _, basis_channels = get_basis_channels_from_t(model, [args.compression] * num_conv)
    print(basis_channels)

    # Create a multitask model with the basis channels estimated above
    mt_model = models.__dict__[args.arch + '_multitask'](basis_channels_list=basis_channels,
        add_bn_prev_list=args.add_bn_prev, add_bn_next_list=args.add_bn_next, carry_all_list=args.carry_all, num_classes=args.increments[0])
    # Initilize the task 1 parameters of multitask model using the weights of conv2d model
    # print(mt_model)
    mt_model.cuda()
    mt_model.load_t1_weights(model)
    # display_stats(mt_model, model, args.jobid + '_' + args.arch, [3, 32, 32], len(args.increments))

    ###########################################################################
    ##################### Finetune Conv Model on Task 1 #######################
    ###########################################################################

    if exec_block(args, '0_ft'):
        if args.compression < 1.0 or args.add_bn_prev: # Finetune

            logger = get_logger(fname='task0_ft', comment='Finetune task 0')
            print('\n\n###########################################\n')
            print('Finetuning task 0')
            [epochs, schedule, lr, weight_decay] = args.epochs, args.schedule, args.lr, args.weight_decay
            args.epochs, args.schedule, args.lr, args.weight_decay = args.ft_epochs, args.ft_schedule, args.ft_lr, args.ft_weight_decay

            optimizer = optim.SGD(mt_model.get_task_parameters(0), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

            mt_model = training_loop_multitask(model=mt_model, optimizer=optimizer, task_id=0, train_loaders=train_loaders,
                                            test_loaders=test_loaders, logger=logger, args=args,
                                            save_best=True)

            [args.epochs, args.schedule, args.lr, args.weight_decay] = epochs, schedule, lr, weight_decay

    else:
        print('Resuming task: 0_ft')
        mt_model = resume(args, '0_ft', mt_model)

    class_incrimental_accuracy = []
    task_prediction_accuracy = []
    tmp, tmp1 = inference(test_loaders[:1], mt_model, args, 1)
    class_incrimental_accuracy.append(tmp)
    task_prediction_accuracy.append(tmp1)

    ###################################################################################################################
    ######################################### Train remaining tasks ###################################################
    ###################################################################################################################

    for i in range(1, len(args.increments)):

        # Add new task parameters to the model
        mt_model.add_task(copy_from=0, growth_rate=args.growth_rate, add_bn_prev_list=args.add_bn_prev, add_bn_next_list=args.add_bn_next, num_classes=args.increments[i])
        mt_model.set_task_id(i)
        mt_model.cuda()

        logger = get_logger(fname='task' + str(i), comment='Train task ' + str(i))
        if exec_block(args, str(i)):
            print('\n\n'+ '_'*90 +'\n')
            print('Training task: ', i)
            # Training model
            optimizer = optim.SGD(mt_model.get_task_parameters(i), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            mt_model = training_loop_multitask(model=mt_model, optimizer=optimizer, task_id=i, train_loaders=train_loaders, test_loaders=test_loaders, logger=logger, args=args,
                                            save_best=True)

        else:
            print(f'Resuming task: {i}')
            mt_model = resume(args, str(i), mt_model)

        print('\nKeys: ', ['time', 'acc1', 'acc5', 'ce_loss'])
        criterion = nn.CrossEntropyLoss()
        ts_vec = []
        for i in range(i + 1):
            ts = test(test_loaders[i], mt_model, args, criterion, i, ['time', 'acc1', 'acc5', 'ce_loss'])
            ts_vec.append(ts[1])
            print('Testing performance of task ' + str(i) + ': ', ts)
        print('Average Task Incremental Accuracy: ', sum(ts_vec) / len(ts_vec))

        tmp, tmp1 = inference(test_loaders[:i+1], mt_model, args, i+1)
        class_incrimental_accuracy.append(tmp)
        task_prediction_accuracy.append(tmp1)

        print('Class Incrimental Accuracy : ', class_incrimental_accuracy)
        print('Avg. Class Incrimental Accuracy : ', sum(class_incrimental_accuracy) / len(class_incrimental_accuracy))
        print('Task Prediction Accuracy : ', task_prediction_accuracy)

        logger.one_time({'task_accuracy':ts_vec, 'class_incrimental_accuracy':class_incrimental_accuracy, 'task_prediction_accuracy':task_prediction_accuracy})

    print('\n')
    print(args)

if __name__ == '__main__':
    main()