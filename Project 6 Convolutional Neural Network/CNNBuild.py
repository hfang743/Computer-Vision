from __main__ import *
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import sys


def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img, volatile=True), size)).data


class CNNBuild(NN.Module):

    def __init__(self):
        super(CNNBuild, self).__init__()

        self.conv1 = NN.Conv2d(3, 18, 3, 1, 1)
        self.conv2 = NN.Conv2d(18, 128, 5, 1)
        self.pool = NN.MaxPool2d(2, 2, 0)
        self.fc1 = NN.Linear(18 * 15 * 15, 2304)
        self.fc2 = NN.Linear(2304, 2)

    def forward(self, x):
        # print(x.shape)

        # x = resize2d(x, (30, 30))
        # print(x.size())
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.pool(x)

        # print(x.shape)
        try:
            x = x.view(-1, 18 * 15 * 15)
        except:
            print('Size mismatch correct view to have multiple of last 3 parameters', x.shape)
            sys.exit()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        # x = F.log_softmax(x, dim=1)

        return (x)
