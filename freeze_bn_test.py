import torch
import torch.nn as nn

# define your model architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(4, 6, bias=False)
        self.fc2 = nn.Linear(6, 8, bias=False)
        self.fc3 = nn.Linear(8, 2, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def print_w(self):
        print(self.fc1.weight)
        print(self.fc2.weight)
        print(self.fc3.weight)

    def freeze(self):
        self.fc1.weight.requires_grad = False

# create an instance of your model
model = MyModel()
model.freeze()
model.train()
model.print_w()
x = torch.randn(1, 4)
y = torch.randn(1, 2)

optimizer = torch.optim.SGD(model.parameters(), 0.9, 0.9, 5e-4)
y_ = model(x)

loss = torch.sum(y - y_)
loss.backward()
optimizer.step()
model.print_w()
