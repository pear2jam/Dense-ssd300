import torch
import torch.nn as nn
import numpy as np


class VGG(torch.nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.act = nn.SiLU()

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv6 = nn.Conv2d(512, 1024, 3, padding=6, dilation=6)

        self.conv7 = nn.Conv2d(1024, 1024, 1)

    def forward(self, X):
        X = self.act(self.conv1_1(X))
        X = self.act(self.conv1_2(X))
        X = self.pool1(X)
    
        X = self.act(self.conv2_1(X))
        X = self.act(self.conv2_2(X))
        X = self.pool2(X)

        X = self.act(self.conv3_1(X))
        X = self.act(self.conv3_2(X))
        X = self.act(self.conv3_3(X))
        X = self.pool3(X)

        X = self.act(self.conv4_1(X))
        X = self.act(self.conv4_2(X))
        X = self.act(self.conv4_3(X))

        conv4_3_res = X

        X = self.pool4(X)
        X = self.act(self.conv5_1(X))
        X = self.act(self.conv5_2(X))
        X = self.act(self.conv5_3(X))
        X = self.pool5(X)
        X = self.act(self.conv6(X))
        X = self.act(self.conv7(X))
        return X, conv4_3_res


class Auxiliairy_layers(torch.nn.Module):
    def __init__(self):
        super(Auxiliairy_layers, self).__init__()

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        self.act = nn.SiLU()

    def forward(self, X):
        X = self.act(self.conv8_1(X))
        X = self.act(self.conv8_2(X))
        conv8_2_res = X
        X = self.act(self.conv9_1(X))
        X = self.act(self.conv9_2(X))
        conv9_2_res = X
        X = self.act(self.conv10_1(X))
        X = self.act(self.conv10_2(X))
        conv10_2_res = X
        X = self.act(self.conv11_1(X))
        X = self.act(self.conv11_2(X))
        return conv8_2_res, conv9_2_res, conv10_2_res, X


class Detection_boxes(torch.nn.Module):
    def __init__(self):
        super(Detection_boxes, self).__init__()

        self.conv4_3 = nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1)
        self.conv10_2 = nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
        self.conv11_2 = nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)

    def forward(self, map4_3, map7, map8_2, map9_2, map10_2, map11_2):
        batch_size = map4_3.shape[0]

        pred4_3 = (self.conv4_3(map4_3))
        pred7 = (self.conv7(map7))
        pred8_2 = (self.conv8_2(map8_2))
        pred9_2 = (self.conv9_2(map9_2))
        pred10_2 = (self.conv10_2(map10_2))
        pred11_2 = (self.conv11_2(map11_2))

        pred4_3 = pred4_3.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        pred7 = pred7.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        pred8_2 = pred8_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        pred9_2 = pred9_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        pred10_2 = pred10_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        pred11_2 = pred11_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        boxes_pred = torch.cat([pred4_3, pred7, pred8_2, pred9_2, pred10_2, pred11_2], axis=1)
        return boxes_pred


class Classes_regression(torch.nn.Module):
    def __init__(self, n_classes):
        super(Classes_regression, self).__init__()
        self.n_classes = n_classes

        self.conv4_3 = nn.Conv2d(512, 4 * n_classes, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(1024, 6 * n_classes, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(512, 6 * n_classes, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(256, 6 * n_classes, kernel_size=3, padding=1)
        self.conv10_2 = nn.Conv2d(256, 4 * n_classes, kernel_size=3, padding=1)
        self.conv11_2 = nn.Conv2d(256, 4 * n_classes, kernel_size=3, padding=1)

    def forward(self, map4_3, map7, map8_2, map9_2, map10_2, map11_2):
        batch_size = map4_3.shape[0]
        n_classes = self.n_classes

        pred4_3 = (self.conv4_3(map4_3))
        pred7 = (self.conv7(map7))
        pred8_2 = (self.conv8_2(map8_2))
        pred9_2 = (self.conv9_2(map9_2))
        pred10_2 = (self.conv10_2(map10_2))
        pred11_2 = (self.conv11_2(map11_2))

        pred4_3 = pred4_3.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, n_classes)
        pred7 = pred7.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, n_classes)
        pred8_2 = pred8_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, n_classes)
        pred9_2 = pred9_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, n_classes)
        pred10_2 = pred10_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, n_classes)
        pred11_2 = pred11_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, n_classes)

        classes_pred = torch.cat([pred4_3, pred7, pred8_2, pred9_2, pred10_2, pred11_2], axis=1)
        return classes_pred

