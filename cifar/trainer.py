import copy
import os
import torch
import time
import torch.nn as nn
import torch.optim as optim

from utils import VectorAccumulator, accuracy, Progressbar, adjust_learning_rate

mse = nn.MSELoss()
mse.cuda()

def train(trainloader, model, args, optimizer, criterion, task_id, keys, talk=False):
    # print('Training...')
    model.train()
    model.set_task_id(task_id)

    accumulator = VectorAccumulator(keys)
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(Progressbar(trainloader, talk=talk)):
        inputs, targets1 = inputs.cuda(), targets.cuda()
        targets = targets1 - sum(args.increments[:task_id])

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # prec1 = sum(model_pred.squeeze(1) == targets)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        # gt_acc.update(prec1.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        accumulator.update( [(time.time() - end), prec1.item(), prec5.item(), loss.item()])
        end = time.time()

    return accumulator.avg

def test(testloader, model, args, criterion, task_id, keys, talk=False):
    # print('Testing...')
    # switch to evaluate mode
    model.eval()
    model.set_task_id(task_id)

    accumulator = VectorAccumulator(keys)
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(Progressbar(testloader, talk=talk)):
        inputs, targets1 = inputs.cuda(), targets.cuda()
        targets = targets1 - sum(args.increments[:task_id])

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        accumulator.update( [(time.time() - end), prec1.item(), prec5.item(), loss.item()])

        end = time.time()

    return accumulator.avg

def inference(testloaders, model, args, num_tasks, talk=False):
    print('Running inference on model to measure class incremental accuracy ...')
    # switch to evaluate mode
    model.eval()

    total_correct_task_id = 0
    total_correct_class_id = 0
    total_images = 0

    for i in range(num_tasks):
        task_id = i
        testloader = testloaders[i]

        num_correct_task_id = 0
        num_correct_class_id = 0
        num_images = 0
        for batch_idx, (inputs, targets) in enumerate(Progressbar(testloader, talk=talk)):
            inputs, targets1 = inputs.cuda(), targets.cuda()
            targets = targets1 - sum(args.increments[:task_id])

            # compute output
            with torch.no_grad():
                outputs = model.ensemble_forward(inputs)

            correct_task_mask, _, _ = predict_task(task_id, outputs)
            filtered_outputs = outputs[task_id][correct_task_mask]
            filtered_targets = targets[correct_task_mask]
            # measure accuracy and record loss
            maxk = max((1,))
            batch_size = targets.size(0)

            _, pred = filtered_outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct_class_mask = pred.eq(filtered_targets.view(1, -1).expand_as(pred))

            num_correct_task_id += correct_task_mask.sum()
            num_correct_class_id += correct_class_mask.sum()
            num_images += batch_size

        total_correct_task_id += num_correct_task_id
        total_correct_class_id += num_correct_class_id
        total_images += num_images

    task_pred_acc_ = total_correct_task_id / total_images * 100.
    task_acc_ = total_correct_class_id / total_images * 100.


    ## report CI acc. and task-id acc.
    print('Total tasks: {}, CIL Acc: {:.2f}, Task-id Cls. Acc.:  {:.2f}'.format(num_tasks, task_acc_, task_pred_acc_))

    return task_acc_.item(), task_pred_acc_.item()

def predict_task(task_id, preoutputs):
    """
      Task id classification
    """
    joint_entropy_tasks = []

        ## calculate entropy of every branch(task)
    for i, preout_ in enumerate(preoutputs):
        ## with shape [bs, num_cls_taski]
        outputs = torch.exp(preout_)

        ## get entropy -\sum_y y * log(y), with shape [bs]
        joint_entropy = -torch.sum(outputs * torch.log(outputs + 0.0001), dim=1)
        """
            normailzing term for entropy given number of classes
        """
        # if i == 0:
        #     p = 5
        # else:
        #     p = 1
        # joint_entropy /= p
        joint_entropy_tasks.append(joint_entropy)

    ## with shape [bs, num_current_task]
    joint_entropy_tasks = torch.stack(joint_entropy_tasks)
    joint_entropy_tasks = joint_entropy_tasks.transpose(0, 1)

    ## mask to indicate the correct task prediction
    ctask = torch.argmin(joint_entropy_tasks, axis=1) == task_id
    correct = sum(ctask)

    return ctask, correct, joint_entropy_tasks

def testing_loop(model, testloader, args, task_id, keys):
    criterion = nn.CrossEntropyLoss()

    test_stats = test(testloader, model, args, criterion, task_id, keys)
    print('\nTest loss: %.4f \nVal accuracy: %.2f%%' % (test_stats[3], test_stats[1]))

def training_loop_multitask(model, optimizer, task_id, train_loaders, test_loaders, logger, args, save_best=True):
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    model.cuda()

    lr = args.lr

    ###################### Main Loop ########################
    best_acc = 0
    best_model = copy.deepcopy(model)
    for epoch in range(args.epochs):
        lr = adjust_learning_rate(optimizer, lr, epoch, args.schedule, args.gamma)
        # print('Training...')
        train_stats = train(train_loaders[task_id], model, args, optimizer, criterion, task_id, logger.keys)

        # print('Testing...')
        test_stats = test(test_loaders[task_id], model, args, criterion, task_id, logger.keys)

        torch.save(model.state_dict(), os.path.join(args.checkpoint, 'model_'+logger.fname.replace('task', '') + '.pth'))

        if best_acc < test_stats[1]:
            best_acc = test_stats[1]
            best_model = copy.deepcopy(model)
            if save_best:
                torch.save(model.state_dict(), os.path.join(args.checkpoint, 'model_'+logger.fname.replace('task', '')+'_best.pth'))

        logger.append([lr, train_stats, test_stats])

        if epoch % args.display_gap == 0:
            print('\nTask ID: [%d] - Epoch: [%d | %d] LR: %f' % (task_id, epoch + 1, args.epochs, lr))
            print('\nKeys: ', logger.keys)
            print('Training: ', train_stats)
            print('Testing: ', test_stats)
            print('Best Acc: ', best_acc)

    print('\nKeys: ', logger.keys)
    ts_vec = []
    for i in range(task_id + 1):
        ts = test(test_loaders[i], best_model, args, criterion, i, logger.keys)
        ts_vec.append(ts[1])
        print('Testing performance of task '+str(i)+': ', ts)
    print('Average Task Incremental Accuracy: ', sum(ts_vec) / len(ts_vec))

    return best_model

def debug_model(model, model_copy):
    input = torch.randn(64, 3, 32, 32)
    input = input.cuda()

    # model_out = model.features[0](input)
    # modelc_out = model_copy.features[0](input)
    # print((model_out - modelc_out).abs().sum())

def stop_running_stats(model):
    for module in model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm2d):
            # if hasattr(module, 'weight'):
            #     module.weight.requires_grad_(False)
            # if hasattr(module, 'bias'):
            #     module.bias.requires_grad_(False)
            module.track_running_stats = False

def moniter_parameters(model, model_copy):

    for mp,cp in zip(model.named_parameters(), model_copy.named_parameters()):
        diff = (mp[1]-cp[1]).abs().sum()
        # st = 'Chnaged      -      ' if diff > 0.01 else 'UNchanged'
        if diff > 0.0001:
            print('Changed - ', mp[0])

def measure_flops(model):
    input = torch.randn(1, 3, 32, 32).to(list(model.parameters())[0].device)
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                model.ensemble_forward(input)
    org_flops = 0
    for ka in prof.key_averages():
        org_flops += ka.flops

    return org_flops