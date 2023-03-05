import torch.nn.functional as F
import torch.nn as nn
import torch
class MySmallCNN(nn.Module):
    def __init__(self):
        super(MySmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1, padding=2)
        self.linear = nn.Linear(in_features=12*8*8, out_features=10) # 10 output classes

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 12*8*8)
        x = self.linear(x)
        return x

from multitask_helpers import MultiTaskModel
class SmallCNNMultitask(MySmallCNN, MultiTaskModel):
    def __init__(self, basis_channels_list, add_bn):
        super().__init__()

        del self.linear
        self.task_id = 0
        self.classifiers = nn.ModuleList()
        self.classifiers.append(nn.Linear(12*8*8, 10))

        self.freeze_preexisitng_bn()
        self.replace_conv2d_with_basisconv2d(basis_channels_list, add_bn)

    def load_t1_weights(self, t1_model):
        self.load_state_dict(t1_model.state_dict(), strict=False)
        self.classifiers[0].weight.data = t1_model.linear.weight.data.clone()
        self.classifiers[0].bias.data = t1_model.linear.bias.data.clone()
        super().load_t1_weights(t1_model)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 12*8*8)

        x = self.classifiers[self.task_id](x)

        return x

def testcnn(num_classes):
    return MySmallCNN()
def testcnn_multitask(basis_channels_list, add_bn, num_classes):
    return SmallCNNMultitask(basis_channels_list, add_bn)