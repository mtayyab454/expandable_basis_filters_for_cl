import torch
from cifar.models.resnet import resnet32_multitask

model = resnet32_multitask(10)
x = torch.randn(1, 3, 32, 32)
print(model(x))