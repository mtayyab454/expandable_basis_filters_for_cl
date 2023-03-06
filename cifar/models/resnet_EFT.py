'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

########################################################################################################################

from multitask_model import MultiTaskModel

class ResNetMultitask(ResNet, MultiTaskModel):
    def __init__(self, basis_channels_list, add_bn_prev_list, add_bn_next_list, block, num_blocks, num_classes):
        super().__init__(block, num_blocks)

        del self.linear
        self.task_id = 0
        self.classifiers = nn.ModuleList()
        self.classifiers.append(nn.Linear(512 * block.expansion, num_classes))

        self.replace_conv2d_with_basisconv2d(basis_channels_list, add_bn_prev_list, add_bn_next_list)

    def load_t1_weights(self, t1_model):
        self.load_state_dict(t1_model.state_dict(), strict=False)
        self.classifiers[0].weight.data = t1_model.linear.weight.data.clone()
        self.classifiers[0].bias.data = t1_model.linear.bias.data.clone()
        super().load_t1_weights(t1_model)

    def set_task_id(self, id):
        self.task_id = id
        super().set_task_id(id)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)

        out = self.classifiers[self.task_id](out)

        return out

def resnet18_multitask(basis_channels_list, add_bn_prev_list, add_bn_next_list, num_classes):
    """ return a ResNet 18 object
    """
    return ResNetMultitask(basis_channels_list, add_bn_prev_list, add_bn_next_list, BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


if __name__ == '__main__':
    # Create a model and train on task 1
    model = resnet18(10)
    model.eval()
    input_tensor = torch.randn(1, 3, 32, 32)
    model_output = model(input_tensor)
    from multitask_model import trace_model, get_basis_channels_from_t
    # Get optimal number of filters for each conv2d layer in the model trained on task 1
    num_conv, num_linear, in_channels, out_channels, basis_channels, layer_type = trace_model(model)
    _, _, basis_channels = get_basis_channels_from_t(model, [1.0]*num_conv)

    # Create a multitask model with the basis channels estimated above
    basis_model = resnet18_multitask(basis_channels, [False]*len(basis_channels), [False]*len(basis_channels), 10)

    # Initilize the task 1 parameters of multitask model using the weights of conv2d model
    basis_model.load_t1_weights(model)
    basis_model.eval()

    # basis_model_output = basis_model(input_tensor)
    basis_model.add_task(0, False, False, 10)
    basis_model.set_task_id(1)
    basis_model_output = basis_model(input_tensor)

    # Check if the results are same.
    print( (model_output-basis_model_output).abs().sum() )
    print(torch.allclose(model_output, basis_model_output, atol=1e-2))  # True
