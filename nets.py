import torch
import torch.nn as nn

class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,10,(3,3),1,1), #N 10 12 12
            nn.BatchNorm2d(10),
            nn.PReLU(),
            nn.MaxPool2d(3,2),#N 10 5 5
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 16,(3,3), 1), #N 16 3 3
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32,(3,3), 1), #N 32 1 1
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)

    def forward(self,x):
        y = self.conv3(self.conv2(self.conv1(x)))
        cls = torch.sigmoid(self.conv4_1(y))
        offset = self.conv4_2(y)
        # landmark = self.conv4_3(y)
        return cls,offset

class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 28, (3,3), 1,1),#N 28 48 48
            nn.BatchNorm2d(28),
            nn.PReLU(),
            nn.MaxPool2d(3, 2 ),#N 28 11 11
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(28, 48, (3,3), 1),#N 48 9 9
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.MaxPool2d(3,2),#N 48 4 4
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, (2,2), 1), #N 64 3 3
            nn.BatchNorm2d(64),
            nn.PReLU())
        self.linear = nn.Linear(64 * 3 * 3, 128)
        self.prlu = nn.PReLU()

        self.linear_1 = nn.Linear(128, 1)
        self.linear_2 = nn.Linear(128, 4)
        self.linear_3 = nn.Linear(128, 10)

    def forward(self, x):
        y = self.conv3(self.conv2(self.conv1(x)))
        y = torch.reshape(y, [y.size(0), -1])
        y = self.linear(y)
        y = self.prlu(y)
        cls = torch.sigmoid(self.linear_1(y))
        offset = self.linear_2(y)
        # landmark = self.linear_3(y)
        return cls, offset

class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32,(3,3), 1 ,1),#N 32 48 48
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),#N 32 23 23
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (3,3), 1),#N 64 21 21
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),#N 64 10 10
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64,(3,3), 1),#N 64 8 8
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),#N 64 4 4
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, (2,2), 1),#N 128 3 3
            nn.BatchNorm2d(128),
            nn.PReLU())
        self.linear = nn.Linear(128 * 3 * 3, 256)
        self.prlu = nn.PReLU()

        self.linear_1 = nn.Linear(256, 1)
        self.linear_2 = nn.Linear(256, 4)
        self.linear_3 = nn.Linear(256, 10)

    def forward(self, x):
        y = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        y = torch.reshape(y, [y.size(0), -1])
        y = self.linear(y)
        cls = torch.sigmoid(self.linear_1(y))
        offset = self.linear_2(y)
        # landmark = self.linear_3(y)
        return cls, offset
