"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def resnet18(num_classes):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

########################################################################################################################

from multitask_model import MultiTaskModel, MultitaskConv2d, fuse_conv_and_bn

class ResNetMultitask(ResNet, MultiTaskModel):
    def __init__(self, basis_channels_list, add_bn_prev_list, add_bn_next_list, carry_all_list, block, num_blocks, num_classes):
        super().__init__(block, num_blocks)

        del self.fc
        self.task_id = 0
        self.classifiers = nn.ModuleList()
        self.classifiers.append(nn.Linear(512 * block.expansion, num_classes))

        self.replace_bn_with_sequential()
        self.replace_conv2d_with_basisconv2d(basis_channels_list, add_bn_prev_list, add_bn_next_list, carry_all_list)

    def load_t1_weights(self, t1_model):
        assert len(self.classifiers) == 1, 'This fucntion should only be called when model has just one task'

        self.load_state_dict(t1_model.state_dict(), strict=False)
        self.classifiers[0].weight.data = t1_model.fc.weight.data.clone()
        self.classifiers[0].bias.data = t1_model.fc.bias.data.clone()

        # Find all MultitaskConv2d in basis_model
        basis_modules = []
        for basis_module in self.modules():
            if isinstance(basis_module, MultitaskConv2d):
                basis_modules.append(basis_module)

        # Find all Conv2d in conv_model
        conv_modules = []
        for conv_module in t1_model.modules():
            if isinstance(conv_module, nn.Conv2d):
                conv_modules.append(conv_module)

        # Find all Batchnorm2d in conv_model
        bn_modules = []
        for bn_module in t1_model.modules():
            if isinstance(bn_module, nn.BatchNorm2d):
                bn_modules.append(bn_module)

        for basis_module, conv_module, bn_module in zip(basis_modules, conv_modules, bn_modules):
            # fused_conv = fuse_conv_and_bn(conv_module, bn_module)
            basis_module.init_weights_from_conv2d(conv_module)
            basis_module.conv_task[0].bn_next.load_state_dict(bn_module.state_dict())

    def set_task_id(self, id):
        self.task_id = id
        super().set_task_id(id)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        # out = self.linear(out)

        output = self.classifiers[self.task_id](output)

        return output

def resnet18_multitask(basis_channels_list, add_bn_prev_list, add_bn_next_list, carry_all_list, num_classes):
    """ return a ResNet 18 object
    """
    return ResNetMultitask(basis_channels_list, add_bn_prev_list, add_bn_next_list, carry_all_list, BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# if __name__ == '__main__':
#     # Create a model and train on task 1
#     model = resnet18(10)
#     model.eval()
#     input_tensor = torch.randn(1, 3, 32, 32)
#     model_output = model(input_tensor)
#     from multitask_model import trace_model, get_basis_channels_from_t
#     # Get optimal number of filters for each conv2d layer in the model trained on task 1
#     num_conv, num_linear, in_channels, out_channels, basis_channels, layer_type = trace_model(model)
#     _, _, basis_channels = get_basis_channels_from_t(model, [1.0]*num_conv)
#
#     # Create a multitask model with the basis channels estimated above
#     basis_model = resnet18_multitask(basis_channels, [False]*len(basis_channels), [False]*len(basis_channels), 10)
#
#     # Initilize the task 1 parameters of multitask model using the weights of conv2d model
#     basis_model.load_t1_weights(model)
#     basis_model.eval()
#
#     # basis_model_output = basis_model(input_tensor)
#     basis_model.add_task(0, False, False, 10)
#     basis_model.set_task_id(1)
#     basis_model_output = basis_model(input_tensor)
#
#     # Check if the results are same.
#     print( (model_output-basis_model_output).abs().sum() )
#     print(torch.allclose(model_output, basis_model_output, atol=1e-2))  # True
