import torch
import torch.nn as nn

from multitask_layer import MultitaskConv2d

def measure_flops(model, input_size):
    input = torch.randn(1, *input_size).to(list(model.parameters())[0].device)
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                # model.ensemble_forward(input)
                model(input)
    flops = 0
    for ka in prof.key_averages():
        flops += ka.flops

    return flops

def count_parameters(basis_model, num_tasks):
    shared_params = []
    task_params = []

    for name, module in basis_model.named_modules():
        if isinstance(module, MultitaskConv2d):
            shared_params.extend(module.conv_shared.parameters())
            task_params.extend(module.conv_task[0].parameters())

    # parameter.extend([p.data for p in self.classifiers[task_id].parameters()])
    task_params.extend( basis_model.classifiers[0].parameters() )

    num_shared_params = sum(p.numel() for p in shared_params)
    num_task_params = sum(p.numel() for p in task_params)*num_tasks

    return (num_shared_params+num_task_params)

def display_stats(basis_model, model, exp_name, input_size, num_tasks):
    org_flops = measure_flops(model, input_size)
    basis_flops = measure_flops(basis_model, input_size)
    num_model_param = sum(p.numel() for p in model.parameters())
    num_basis_param = count_parameters(basis_model, num_tasks)

    print_text = f"\n############################################# {exp_name} #############################################\n"
    print_text += f"\n    Model FLOPs: {org_flops / 10 ** 6:.2f}M"
    print_text += f"\n    Basis Model FLOPs: {basis_flops / 10 ** 6:.2f}M"
    print_text += f"\n    % Reduction in FLOPs: {100 - (basis_flops * 100 / org_flops):.2f} %"
    print_text += f"\n    % Speedup: {org_flops / basis_flops:.2f} %\n"

    print_text += f"\n    Model Total Params: {num_model_param / 10 ** 6:.2f}M"
    print_text += f"\n    Basis Model Total params: {num_basis_param / 10 ** 6:.2f}M"
    print_text += f"\n    % Reduction in Total params: {100 - (num_basis_param * 100 / num_model_param):.2f} %\n"

    print(print_text)


def trace_model(model):
    in_channels, out_channels, basis_channels, layer_type = [], [], [], []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            in_channels.append(module.in_channels)
            out_channels.append(module.out_channels)
            basis_channels.append(
                min(module.out_channels, module.in_channels * module.kernel_size[0] * module.kernel_size[1]))
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

            basis_channels.append(min(idx + 1, module.in_channels * module.kernel_size[0] * module.kernel_size[1]))

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
            if isinstance(add_bn_prev_list, list):
                add_bn_prev = add_bn_prev_list.pop(0)
            else:
                add_bn_prev = add_bn_prev_list

            if isinstance(add_bn_next_list, list):
                add_bn_next = add_bn_next_list.pop(0)
            else:
                add_bn_next = add_bn_next_list

            in_channels = child_module.in_channels
            basis_channels = basis_channels_list.pop(0)

            out_channels = child_module.out_channels
            kernel_size = child_module.kernel_size
            stride = child_module.stride
            padding = child_module.padding
            dilation = child_module.dilation
            groups = child_module.groups

            basis_layer = MultitaskConv2d(add_bn_prev, add_bn_next, in_channels, basis_channels, out_channels,
                                          kernel_size, stride, padding, dilation, groups)
            setattr(module, name, basis_layer)
            # module._modules[name] = basis_layer
        else:
            # Recursively apply the function to the child module
            _replace_conv2d_with_basisconv2d(child_module, basis_channels_list, add_bn_prev_list, add_bn_next_list)

def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer

def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)

def fuse_conv_and_bn(conv, bn):
    #
    # init
    fusedconv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )
    fusedconv.to(conv.weight.device)
    #
    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.data.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))
    #
    # prepare spatial bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.weight.size(0))
        b_conv = b_conv.to(conv.weight.device)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.data.copy_(torch.matmul(w_bn, b_conv) + b_bn)
    #
    # we're done
    return fusedconv

class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        # print('MultiTaskModel')

    def replace_conv2d_with_basisconv2d(self, basis_channels_list, add_bn_prev_list, add_bn_next_list):
        _replace_conv2d_with_basisconv2d(self, basis_channels_list, add_bn_prev_list, add_bn_next_list)

    def replace_bn_with_sequential(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                set_layer(self, name, nn.Sequential())
        # print('done')

    def set_task_id(self, id):
        for name, module in self.named_modules():
            if isinstance(module, MultitaskConv2d):
                module.set_task_id(id)

    def add_task(self, copy_from, growth_rate, add_bn_prev_list, add_bn_next_list, num_classes):

        ln = nn.Linear(self.classifiers[0].weight.shape[1], num_classes)
        ln.load_state_dict(self.classifiers[copy_from].state_dict())
        self.classifiers.append(ln)

        for name, module in self.named_modules():
            if isinstance(module, MultitaskConv2d):
                if isinstance(add_bn_prev_list, list):
                    add_bn_prev = add_bn_prev_list.pop(0)
                else:
                    add_bn_prev = add_bn_prev_list

                if isinstance(add_bn_next_list, list):
                    add_bn_next = add_bn_next_list.pop(0)
                else:
                    add_bn_next = add_bn_next_list
                module.add_task(copy_from, growth_rate, add_bn_prev, add_bn_next)

    def get_task_parameter(self, task_id):
        parameter = []

        for name, module in self.named_modules():
            if isinstance(module, MultitaskConv2d):
                # parameter.extend([p.data for p in module.conv_task[task_id].parameters()])
                parameter.extend(module.conv_task[task_id].parameters())
                parameter.extend(module.shared_weights)

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
