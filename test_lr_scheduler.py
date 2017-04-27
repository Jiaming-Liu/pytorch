import unittest
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from lr_scheduler import *


class Net(nn.Module):  # copied from tutorial
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.net = Net()
        self.opt = SGD(self.net.parameters(), lr=0.05)
        # For testing with multiple param groups.
        self.opt.param_groups.append({'lr': 0})

    def test_step_lr(self):
        # lr = 0.05     if epoch < 3
        # lr = 0.005    if 30 <= epoch < 6
        # lr = 0.0005   if epoch >= 9
        targets = [[0.05] * 3 + [0.005] * 3 + [0.0005] * 3 + [0.00005] * 3]
        scheduler = StepLR(self.opt, base_lr=0.05, gamma=0.1, step_size=3)
        epochs = 10
        self._test(scheduler, targets, epochs)

    def test_multi_step_lr(self):
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if epoch < 9
        # lr = 0.00005   if epoch >= 9
        targets = [[0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 3]
        scheduler = MultiStepLR(self.opt, base_lr=0.05, gamma=0.1, milestones=[2, 5, 9])
        epochs = 10
        self._test(scheduler, targets, epochs)

    def test_exp_lr(self):
        targets = [[0.05 * (0.9 ** x) for x in range(10)]]
        scheduler = ExponentialLR(self.opt, base_lr=0.05, gamma=0.9)
        epochs = 10
        self._test(scheduler, targets, epochs)

    def test_group_lambda_lr(self):
        targets = [[0.05 * (0.9 ** x) for x in range(10)], [0.4 * (0.8 ** x) for x in range(10)]]
        scheduler = GroupLambdaLR(self.opt, base_lrs=[0.05, 0.4], lr_lambdas=[lambda x1: 0.9 ** x1, lambda x2: 0.8 ** x2])
        epochs = 10
        self._test(scheduler, targets, epochs)

    def test_reduce_lr_on_plateau1(self):
        for param_group in self.opt.param_groups:
            param_group['lr'] = 0.5
        targets = [ [0.5] * 20 ]
        metrics = range(20,-1,-1)
        scheduler = ReduceLROnPlateau(self.opt)
        epochs = 10
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau2(self):
        for param_group in self.opt.param_groups:
            param_group['lr'] = 0.5
        targets = [ [0.5] * (2+6) + [0.05] * (5+5) +  [0.005] * 4]
        metrics = [0.8] * 2 + [0.234] * 20
        scheduler = ReduceLROnPlateau(self.opt,patience=5,cooldown=5)
        epochs = 22
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau3(self):
        for param_group in self.opt.param_groups:
            param_group['lr'] = 0.5
        targets = [ [0.5] * (2+6) + [0.05] * (5+5) +  [0.005] * 4]
        metrics = [-0.8] * 2 + [-0.234] * 20
        scheduler = ReduceLROnPlateau(self.opt,mode='max',patience=5,cooldown=5)
        epochs = 22
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def _test(self, scheduler, targets, epochs=10):
        for epoch in range(epochs):
            scheduler.step(epoch)
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertAlmostEquals(target[epoch], param_group['lr'], msg='LR is wrong in epoch {}'.format(epoch))

    def _test_reduce_lr_on_plateau(self, scheduler, targets, metrics, epochs=10, verbose=0):
        for epoch in range(epochs):
            scheduler.step(epoch,metrics[epoch])
            if verbose>0:
                print('epoch{}:\tlr={}'.format(epoch,self.opt.param_groups[0]['lr']))
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertAlmostEquals(target[epoch], param_group['lr'], msg='LR is wrong in epoch {}'.format(epoch))


if __name__ == '__main__':
    unittest.main()
