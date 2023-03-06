import torch
import torch.nn as nn

# define your model architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc(x)
        return x

# create an instance of your model
model = MyModel()

# freeze the weights of BN layers
for name, param in model.named_parameters():
    if 'bn' in name:
        param.requires_grad = False

net = nn.Sequential()
x = torch.randn(1, 3, 32, 32)
y = net(x)

print(y-x)