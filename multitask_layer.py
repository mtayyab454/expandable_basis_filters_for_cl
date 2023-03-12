import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

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

class MultitaskConv2d(nn.Conv2d):
    def __init__(self, add_bn_prev, add_bn_next, in_channels, basis_channels, out_channels, kernel_size, stride, padding, dilation, groups):
        super(MultitaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)

        self.task_id = 0
        # define new convolution layers with F and w
        # self.conv_shared = nn.Conv2d(in_channels, basis_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.shared_weights = nn.ParameterList([nn.Parameter(torch.Tensor(basis_channels, in_channels, *kernel_size))])
        tc = TaskConv2d(add_bn_prev=add_bn_prev, add_bn_next=add_bn_next, in_channels=basis_channels, out_channels=out_channels)

        self.conv_task = nn.ModuleList()
        self.conv_task.append(tc)

    def init_weights_from_conv2d(self, conv2d, sparse_filters=False):
        weight = conv2d.weight.data.clone()
        bias = conv2d.bias.data.clone() if conv2d.bias is not None else None
        assert self.shared_weights[0].shape[0] <= weight.numel() // weight.size(0), "Number of filters should be less than or match input tensor dimensions"
        # apply SVD to get F and w
        F, w = self.svd_init(weight, sparse_filters)

        # Set the weights of the new convolution layers.
        self.shared_weights[0].data = F.view(self.shared_weights[0].shape[0], *weight.shape[1:] ).to(weight.dtype)
        self.conv_task[0].conv_t.weight.data = w.unsqueeze(-1).unsqueeze(-1).to(weight.dtype)

        if bias is not None:
            self.conv_task[0].conv_t.bias.data = bias.to(bias.dtype)
        else:
            self.conv_task[0].conv_t.bias.data.zero_()

    def set_task_id(self, task_id):
        self.task_id = task_id
        for w in self.shared_weights:
            w.requires_grad = False

        self.shared_weights[self.task_id].requires_grad = True

    def train(self, mode=True):
        super().train()
        for w in self.shared_weights:
            w.requires_grad = False

        self.shared_weights[self.task_id].requires_grad = True

    def add_task(self, copy_from, growth_rate, add_bn_prev, add_bn_next):
        assert growth_rate >= 0.0 and growth_rate <= 1.0, 'Growth rate must lie in [0-1]'

        basis_channels = self.shared_weights[0].shape[0]
        sfn = math.ceil(basis_channels*growth_rate)
        sc = nn.Parameter(torch.Tensor(sfn, self.in_channels, *self.kernel_size))
        torch.nn.init.kaiming_uniform_(sc, a=math.sqrt(5))

        sc.requires_grad = False
        self.shared_weights.append(sc)

        task_in_ch = [w.shape[0] for w in self.shared_weights]

        tc = TaskConv2d(add_bn_prev=add_bn_prev, add_bn_next=add_bn_next,
                        in_channels=sum(task_in_ch), out_channels=self.conv_task[0].conv_t.out_channels)

        # tc.load_state_dict(self.conv_task[copy_from].state_dict(), strict=False)
        self.conv_task.append(tc)

    def svd_init(self, weight, sparse_filters):

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


        F = v_t[:, 0:self.shared_weights[0].shape[0]].t()
        w = torch.mm(F, H.t()).t()

        return F, w

    def forward(self, x):
        # convolve with F and then w

        # x_ = [F.conv2d(x, self.shared_weights[i], None, self.stride, self.padding, self.dilation, self.groups) for i in range(self.task_id+1)]
        # x = torch.cat(x_, dim=1)

        weights = torch.cat([self.shared_weights[i] for i in range(self.task_id+1)], dim=0)
        x = F.conv2d(x, weights, None, self.stride, self.padding, self.dilation, self.groups)

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
        False,
        True,
        conv.in_channels,
        16,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
    )
    basis_conv.init_weights_from_conv2d(conv)
    basis_conv.eval()

    # perform forward pass with both Conv2d and BasisConv2d modules
    y_conv = conv(x)
    y_basis = basis_conv(x)
    print(torch.allclose(y_conv, y_basis, atol=1e-5))  # True
    optimizer = optim.SGD()

    basis_conv.add_task(copy_from=0, growth_rate=0.5, add_bn_prev=False, add_bn_next=True)
    basis_conv.set_task_id(1)
    y_basis = basis_conv(x)

    loss = (y_conv - y_basis).abs().sum()
    print(loss)
    loss.backward()