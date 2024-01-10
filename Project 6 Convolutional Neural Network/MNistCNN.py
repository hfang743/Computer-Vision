from __main__ import *


class MNistCNN(NN.Module):

    def __init__(self):
        super(MNistCNN, self).__init__()
        self.conv1 = NN.Conv2d(1, 20, 5, 1)
        self.conv2 = NN.Conv2d(20, 50, 5, 1)
        self.fc1 = NN.Linear(800, 400)
        self.fc2 = NN.Linear(400, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 800)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

