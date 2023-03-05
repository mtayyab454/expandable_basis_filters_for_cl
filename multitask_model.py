import copy

import torch
import torch.nn as nn

from multitask_layer import MultitaskConv2d

def trace_model(model):
    in_channels, out_channels, basis_channels, layer_type = [], [], [], []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            in_channels.append(module.in_channels)
            out_channels.append(module.out_channels)
            basis_channels.append(min(module.out_channels, module.in_channels * module.kernel_size[0] * module.kernel_size[1]))
            layer_type.append('conv')
        elif isinstance(module, nn.Linear):
            in_channels.append(module.in_features)
            out_channels.append(module.out_features)
            basis_channels.append(min(module.out_features, module.in_features))
            layer_type.append('linear')

    num_conv = sum(1 for lt in layer_type if lt == 'conv')
    num_linear = sum(1 for lt in layer_type if lt == 'linear')

    return num_conv, num_linear, in_channels, out_channels, basis_channels, layer_type
def get_basis_channels_from_t(model, t):

    assert all(0 <= x <= 1 for x in t), "Values of t must be between 0 and 1"

    in_channels, out_channels, basis_channels = [], [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):

            in_channels.append(module.in_channels)
            out_channels.append(module.out_channels)

            weight = module.weight.data.clone()
            H = weight.view(weight.shape[0], -1)
            [u, s, v_t] = torch.svd(H)
            _, ind = torch.sort(s, descending=True)
            delta = s[ind] ** 2

            c_sum = torch.cumsum(delta, 0)
            c_sum = c_sum / c_sum[-1]
            idx = torch.nonzero(c_sum >= t.pop(0))[0].item()

            basis_channels.append(min(idx+1, module.in_channels * module.kernel_size[0] * module.kernel_size[1]))

    return in_channels, out_channels, basis_channels

def _replace_conv2d_with_basisconv2d(module, basis_channels_list, add_bn_prev_list, add_bn_next_list):
    """
    Recursively replaces all Conv2d layers in a module with BasisConv2d layers.

    Args:
        module (torch.nn.Module): The module whose Conv2d layers will be replaced.
        basis_channels_list (list, optional): A list of integers specifying the number
            of basis channels for each BasisConv2d layer in the model. If None, the number
            of basis channels will be set to min(out_channels, weight.numel() // weight.size(0)).
        add_bn_list (list, optional): A list of Boolean values specifying whether to add
            batch normalization to each BasisConv2d layer in the model. If None, batch
            normalization will not be added.

    Returns:
        None.
    """
    for name, child_module in module.named_children():
        if isinstance(child_module, nn.Conv2d):
            # Replace the Conv2d layer with a BasisConv2d layer
            # weight = child_module.weight.data.clone()
            # bias = child_module.bias.data.clone() if child_module.bias is not None else None
            add_bn_prev = add_bn_prev_list.pop(0)
            add_bn_next = add_bn_next_list.pop(0)

            in_channels = child_module.in_channels
            basis_channels = basis_channels_list.pop(0)

            out_channels = child_module.out_channels
            kernel_size = child_module.kernel_size
            stride = child_module.stride
            padding = child_module.padding
            dilation = child_module.dilation
            groups = child_module.groups

            basis_layer = MultitaskConv2d(add_bn_prev, add_bn_next, in_channels, basis_channels, out_channels, kernel_size, stride, padding, dilation, groups)
            setattr(module, name, basis_layer)
            # module._modules[name] = basis_layer
        else:
            # Recursively apply the function to the child module
            _replace_conv2d_with_basisconv2d(child_module, basis_channels_list, add_bn_prev_list, add_bn_next_list)

def _freeze_preexisitng_bn(parent_module):

    for name, module in parent_module.named_children():

        if isinstance(module, nn.BatchNorm2d):
            # print(name)
            module.track_running_stats = False
            for param in module.parameters():
                param.requires_grad = False
        elif not isinstance(module, MultitaskConv2d):
            _freeze_preexisitng_bn(module)

# def _load_t1_weights(basis_module, conv_module):


class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        # print('MultiTaskModel')

    def replace_conv2d_with_basisconv2d(self, basis_channels_list, add_bn_prev_list, add_bn_next_list):
        _replace_conv2d_with_basisconv2d(self, basis_channels_list, add_bn_prev_list, add_bn_next_list)

    def freeze_preexisitng_bn(self):
        # Freeze preexisting BatchNorm2d layers
        _freeze_preexisitng_bn(self)

    def load_t1_weights(self, t1_model):

        self.load_state_dict(t1_model.state_dict(), strict=False)
        self.classifiers[0].weight.data = t1_model.linear.weight.data.clone()
        self.classifiers[0].bias.data = t1_model.linear.bias.data.clone()

        basis_modules = []
        for basis_module in self.modules():
            if isinstance(basis_module, MultitaskConv2d):
                basis_modules.append(basis_module)

        conv_modules = []
        for conv_module in t1_model.modules():
            if isinstance(conv_module, nn.Conv2d):
                conv_modules.append(conv_module)

        for basis_module, conv_module in zip(basis_modules, conv_modules):
            basis_module.init_weights_from_conv2d(conv_module)

    def set_task_id(self, id):
        for name, module in self.named_modules():
            if isinstance(module, MultitaskConv2d):
                module.task_id = id

    def add_task(self, classifier_in, copy_from, num_classes):

        ln = nn.Linear(classifier_in, num_classes)
        ln.weight.data = self.classifiers[copy_from].weight.data.clone()
        ln.bias.data = self.classifiers[copy_from].bias.data.clone()
        self.classifiers.append(ln)

        for name, module in self.named_modules():
            if isinstance(module, MultitaskConv2d):
                module.add_task(copy_from)

    def get_task_parameter(self, task_id):
        parameter = []

        for name, module in self.named_modules():
            if isinstance(module, MultitaskConv2d):
                # parameter.extend([p.data for p in module.conv_task[task_id].parameters()])
                parameter.extend(module.conv_task[task_id].parameters())
                if task_id == 0:
                    # parameter.extend([p.data for p in module.conv_shared.parameters()])
                    parameter.extend(module.conv_shared.parameters())

        # parameter.extend([p.data for p in self.classifiers[task_id].parameters()])
        parameter.extend(self.classifiers[task_id].parameters())

        return parameter

    def ensemble_forward(self, x):
        outputs = []
        for i in range(len(self.classifiers)):
            self.set_task_id(i)
            out = self.forward(x)
            outputs.append(out)

        return outputs