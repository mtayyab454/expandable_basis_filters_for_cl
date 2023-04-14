import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

class SharedConvList(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, carry_all):
        super(SharedConvList, self).__init__()
        self.task_id = 0
        self.carry_all = carry_all
        self.conv_s = nn.ModuleList()
        self.conv_s.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False))

    def set_task_id(self, task_id):
        self.task_id = task_id

    def get_task_parameters(self, task_id):
        growth = len(self.conv_s) > 1
        parameter = []
        if len(self.conv_s) == 1 and self.task_id == 0: # Growth is 0 and task id is 0
            parameter.extend(self.conv_s[0].parameters())
        elif len(self.conv_s) > 1: # Growth is > 0
            parameter.extend(self.conv_s[task_id].parameters())
        else:
            pass

        return parameter

    def train(self, mode=True):
        for c in self.conv_s:
            c.eval()

        if len(self.conv_s) == 1 and self.task_id == 0: # Growth is 0 and task id is 0
            self.conv_s[0].train(mode)
        elif len(self.conv_s) > 1: # Growth is > 0
            self.conv_s[self.task_id].train(mode)
        else:
            pass

    def add_task(self, copy_from, growth_rate):
        if growth_rate > 0.0:
            basis_channels = self.conv_s[0].out_channels
            sfn = math.ceil(basis_channels*growth_rate)
            sc = nn.Conv2d(self.conv_s[0].in_channels, sfn, self.conv_s[0].kernel_size, self.conv_s[0].stride, self.conv_s[0].padding, self.conv_s[0].dilation, self.conv_s[0].groups, bias=False)

            if copy_from == 0:
                sc.weight.data.copy_(self.conv_s[0].weight.data[0:sfn, :, :, :])
            else:
                sc.weight.data.copy_(self.conv_s[copy_from].weight.data)
            # torch.nn.init.kaiming_uniform_(sc.weight, a=math.sqrt(5))

            self.conv_s.append(sc)

    def forward(self, x):
        if self.task_id == 0:
            out = self.conv_s[0](x)
        else:
            if self.carry_all:
                out = torch.cat([self.conv_s[i](x) for i in range(self.task_id+1)], dim=1)
            else:
                out = torch.cat([self.conv_s[0](x), self.conv_s[self.task_id](x)], dim=1)

        return out

class TaskConv2d(nn.Module):
    def __init__(self, add_bn_prev, add_bn_next, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1):
        super(TaskConv2d, self).__init__()

        self.conv_t = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)

        if add_bn_prev:
            self.bn_prev = nn.BatchNorm2d(in_channels)
            self.bn_prev.weight.data.fill_(1)
            self.bn_prev.bias.data.zero_()
        else:
            self.bn_prev = None

        if add_bn_next:
            self.bn_next = nn.BatchNorm2d(out_channels)
            self.bn_next.weight.data.fill_(1)
            self.bn_next.bias.data.zero_()
        else:
            self.bn_next = None

    def forward(self, x):
        if self.bn_prev is not None:
            x = self.bn_prev(x)

        x = self.conv_t(x)

        if self.bn_next is not None:
            x = self.bn_next(x)
        return x

class MultitaskConv2d(nn.Module):
    def __init__(self, add_bn_prev, add_bn_next, in_channels, basis_channels, out_channels, kernel_size, stride, padding, dilation, groups, carry_all):
        super(MultitaskConv2d, self).__init__()

        self.carry_all = carry_all
        self.task_id = 0
        # self.shared_weights = nn.ParameterList([nn.Parameter(torch.Tensor(basis_channels, in_channels, *kernel_size))])
        self.conv_shared = SharedConvList(in_channels, basis_channels, kernel_size, stride, padding, dilation, groups, carry_all)
        tc = TaskConv2d(add_bn_prev=add_bn_prev, add_bn_next=add_bn_next, in_channels=basis_channels, out_channels=out_channels)

        self.conv_task = nn.ModuleList()
        self.conv_task.append(tc)

    def svd_init(self, weight, basis_channels, sparse_filters):

        H = weight.view(weight.shape[0], -1)

        if sparse_filters:
            H = torch.mm(torch.t(H), H)
            [u, s, v_t] = torch.svd(H, some=False)
            _, ind = torch.sort(s, descending=True)
            delta = s[ind]
            v_t = v_t[:, ind]
        else:
            [u, s, v_t] = torch.svd(H)
            _, ind = torch.sort(s, descending=True)
            delta = s[ind] ** 2
            v_t = v_t[:, ind]


        F = v_t[:, 0:basis_channels].t()
        w = torch.mm(F, H.t()).t()

        return F, w

    def init_weights_from_conv2d(self, conv2d, sparse_filters=False):
        weight = conv2d.weight.data.clone()
        bias = conv2d.bias.data.clone() if conv2d.bias is not None else None
        basis_channels = self.conv_shared.conv_s[0].out_channels

        assert basis_channels <= weight.numel() // weight.size(0), "Number of basis filters should be less than or match input tensor dimensions"
        # apply SVD to get F and w
        F, w = self.svd_init(weight, basis_channels, sparse_filters)

        # Set the weights of the new convolution layers.
        self.conv_shared.conv_s[0].weight.data = F.view(basis_channels, *weight.shape[1:] ).to(weight.dtype)
        self.conv_task[0].conv_t.weight.data = w.unsqueeze(-1).unsqueeze(-1).to(weight.dtype)

        if bias is not None:
            self.conv_task[0].conv_t.bias.data = bias.to(bias.dtype)
        else:
            self.conv_task[0].conv_t.bias.data.zero_()

    def set_task_id(self, task_id):
        self.task_id = task_id
        self.conv_shared.set_task_id(task_id)

    def get_task_parameters(self, task_id):
        parameter = []
        parameter.extend(self.conv_shared.get_task_parameters(task_id))
        parameter.extend(self.conv_task[task_id].parameters())

        return parameter

    def add_task(self, copy_from, growth_rate, add_bn_prev, add_bn_next):
        assert growth_rate >= 0.0 and growth_rate <= 1.0, 'Growth rate must be in the range [0-1]'

        self.conv_shared.add_task(copy_from, growth_rate)

        if self.carry_all:
            task_in_ch = [w.weight.shape[0] for w in self.conv_shared.conv_s]
            tc = TaskConv2d(add_bn_prev=add_bn_prev, add_bn_next=add_bn_next,
                            in_channels=sum(task_in_ch), out_channels=self.conv_task[0].conv_t.out_channels)
        else:
            task_in_ch = [self.conv_shared.conv_s[0].weight.shape[0], self.conv_shared.conv_s[-1].weight.shape[0]]
            tc = TaskConv2d(add_bn_prev=add_bn_prev, add_bn_next=add_bn_next,
                            in_channels=sum(task_in_ch), out_channels=self.conv_task[0].conv_t.out_channels)


        if growth_rate == 0:
            tc.load_state_dict(self.conv_task[copy_from].state_dict(), strict=True)
        else:
            prev_basis_channels = self.conv_task[copy_from].conv_t.weight.shape[1]
            tc.conv_t.weight.data[:,0:prev_basis_channels,:,:].copy_(self.conv_task[copy_from].conv_t.weight.data)
            # torch.nn.init.kaiming_uniform_(tc.conv_t.weight, a=math.sqrt(5))
            # print(f'task layer weights cannot be coppied from task {copy_from} since growth_rate, {growth_rate} is > 0')

        self.conv_task.append(tc)

    def forward(self, x):
        x = self.conv_shared(x)
        x = self.conv_task[self.task_id](x)
        return x

if __name__ == '__main__':
    # create an input tensor
    x = torch.randn(1, 3, 32, 32)

    # create a Conv2d layer with random weights
    conv = torch.nn.Conv2d(3, 16, kernel_size=3, bias=True)
    conv.eval()

    # create a BasisConv2d module using the weights of the Conv2d layer
    basis_conv = MultitaskConv2d(
        add_bn_prev=False,
        add_bn_next=True,
        in_channels=conv.in_channels,
        basis_channels=16,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        carry_all=False)
    basis_conv.init_weights_from_conv2d(conv)

    y_conv = conv(x)

    # test task 0
    print(basis_conv.conv_shared.conv_s[0].weight.sum())
    basis_conv.eval()
    basis_conv.set_task_id(0)
    y_basis = basis_conv(x)
    print(torch.allclose(y_conv, y_basis, atol=1e-5))  # True

    # test task 0
    basis_conv.add_task(copy_from=0, growth_rate=0.1, add_bn_prev=False, add_bn_next=True)
    print(basis_conv.conv_shared.conv_s[0].weight.sum())
    print(basis_conv.conv_shared.conv_s[1].weight.sum())
    basis_conv.eval()
    basis_conv.set_task_id(1)
    y_basis = basis_conv(x)
    print(torch.allclose(y_conv, y_basis, atol=1e-5))  # True

    # import torch.optim as optim
    # optimizer = optim.SGD(basis_conv.get_task_parameters(1), lr=0.1)
    #
    # basis_conv.train()
    #
    # # for name, parameter in basis_conv.named_parameters():
    # #     print(f'{name} is {parameter.requires_grad}')
    # #
    # # for name, module in basis_conv.named_modules():
    # #     print(f'{name} is {module.training}')
    #
    # basis_conv.set_task_id(1)
    # y_basis = basis_conv(x)
    #
    # loss = (y_conv - y_basis).abs().sum()
    # print(loss)
    # loss.backward()
    #
    # optimizer.step()
    #
    # print(basis_conv.conv_shared.conv_s[0].weight.sum())
    # print(basis_conv.conv_shared.conv_s[1].weight.sum())