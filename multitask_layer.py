import torch
import torch.nn as nn
from collections import OrderedDict

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
    def __init__(self, add_bn_prev, add_bn_next, in_channels, basis_channels, out_channels, kernel_size, stride, padding, dilation, groups):
        super(MultitaskConv2d, self).__init__()

        self.task_id = 0
        # define new convolution layers with F and w
        self.conv_shared = nn.Conv2d(in_channels, basis_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        tc = TaskConv2d(add_bn_prev=add_bn_prev, add_bn_next=add_bn_next, in_channels=basis_channels, out_channels=out_channels)

        self.conv_task = nn.ModuleList()
        self.conv_task.append(tc)

    def init_weights_from_conv2d(self, conv2d, sparse_filters=False):
        weight = conv2d.weight.data.clone()
        bias = conv2d.bias.data.clone() if conv2d.bias is not None else None
        assert self.conv_shared.out_channels <= weight.numel() // weight.size(0), "Number of filters should be less than or match input tensor dimensions"
        # apply SVD to get F and w
        F, w = self.svd_init(weight, sparse_filters)

        # Set the weights of the new convolution layers.
        self.conv_shared.weight.data = F.view(self.conv_shared.out_channels, *weight.shape[1:] ).to(weight.dtype)
        self.conv_task[0].conv_t.weight.data = w.unsqueeze(-1).unsqueeze(-1).to(weight.dtype)

        if bias is not None:
            self.conv_task[0].conv_t.bias.data = bias.to(bias.dtype)
        else:
            self.conv_task[0].conv_t.bias.data.zero_()

    def add_task(self, copy_from=0):
        tc = TaskConv2d(add_bn_next=True if self.conv_task[0].bn_next is not None else False, add_bn_prev=False,
                        in_channels=self.conv_task[0].conv_t.in_channels, out_channels=self.conv_task[0].conv_t.out_channels)

        tc.load_state_dict(self.conv_task[copy_from].state_dict())
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


        F = v_t[:, 0:self.conv_shared.out_channels].t()
        w = torch.mm(F, H.t()).t()

        return F, w

    def forward(self, x):
        # convolve with F and then w
        x = self.conv_shared(x)
        x = self.conv_task[self.task_id](x)
        return x

if __name__ == '__main__':
    # create an input tensor
    x = torch.randn(1, 3, 32, 32)

    # create a Conv2d layer with random weights
    conv = torch.nn.Conv2d(3, 16, kernel_size=3, bias=False)
    conv.eval()

    # create a BasisConv2d module using the weights of the Conv2d layer
    basis_conv = MultitaskConv2d(
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

    basis_conv.add_task()
    basis_conv.task_id = 1
    basis_conv.eval()

    # perform forward pass with both Conv2d and BasisConv2d modules
    y_conv = conv(x)
    y_basis = basis_conv(x)
    print(torch.allclose(y_conv, y_basis, atol=1e-5))  # True