import torch
import torch.nn as nn
import torchvision.transforms as tt


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

        pred4_3 = self.conv4_3(map4_3)
        pred7 = self.conv7(map7)
        pred8_2 = self.conv8_2(map8_2)
        pred9_2 = self.conv9_2(map9_2)
        pred10_2 = self.conv10_2(map10_2)
        pred11_2 = self.conv11_2(map11_2)

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

        self.conv4_3 = nn.Conv2d(512, 4*n_classes, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(1024, 6*n_classes, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(512, 6*n_classes, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(256, 6*n_classes, kernel_size=3, padding=1)
        self.conv10_2 = nn.Conv2d(256, 4*n_classes, kernel_size=3, padding=1)
        self.conv11_2 = nn.Conv2d(256, 4*n_classes, kernel_size=3, padding=1)

    def forward(self, map4_3, map7, map8_2, map9_2, map10_2, map11_2):
        batch_size = map4_3.shape[0]
        n_classes = self.n_classes

        pred4_3 = self.conv4_3(map4_3)
        pred7 = self.conv7(map7)
        pred8_2 = self.conv8_2(map8_2)
        pred9_2 = self.conv9_2(map9_2)
        pred10_2 = self.conv10_2(map10_2)
        pred11_2 = self.conv11_2(map11_2)

        pred4_3 = pred4_3.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, n_classes)
        pred7 = pred7.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, n_classes)
        pred8_2 = pred8_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, n_classes)
        pred9_2 = pred9_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, n_classes)
        pred10_2 = pred10_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, n_classes)
        pred11_2 = pred11_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, n_classes)

        classes_pred = torch.cat([pred4_3, pred7, pred8_2, pred9_2, pred10_2, pred11_2], axis = 1)
        return classes_pred


class SSD300(torch.nn.Module):
    def __init__(self, n_classes, dropout_rate=0):
        super(SSD300, self).__init__()

        self.act = nn.ReLU()
        self.dout = nn.Dropout(dropout_rate)
        self.classes_pred = Classes_regression(n_classes)
        self.boxes_pred = Detection_boxes()

        # conv 1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv 3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # conv 4
        self.conv4_1 = nn.Conv2d(256 + 64, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv 5
        self.conv5_1 = nn.Conv2d(512 + 64 + 128, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # conv 6
        self.conv6 = nn.Conv2d(512 + 64 + 128 + 256, 1024, 3, padding=6, dilation=6)

        # conv 7
        self.conv7 = nn.Conv2d(1024 + 64 + 128 + 256 + 512, 1024, 1)

        # Auxiliairy_layers

        # conv 8
        self.conv8_1 = nn.Conv2d(1024 + 64 + 128 + 256 + 512 + 512, 256, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        # conv 9
        self.conv9_1 = nn.Conv2d(512 + 64 + 128 + 256 + 512 + 512 + 1024, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # conv 10
        self.conv10_1 = nn.Conv2d(256 + 64 + 128 + 256 + 512 + 512 + 1024 + 1024, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        # conv 11
        self.conv11_1 = nn.Conv2d(256 + 64 + 128 + 256 + 512 + 512 + 1024 + 1024 + 512, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        self.res1_3 = nn.Conv2d(64, 128, kernel_size=1)
        self.res2_4 = nn.Conv2d(128, 256, kernel_size=1)
        self.res3_5 = nn.Conv2d(256, 512, kernel_size=1)
        self.res4_6 = nn.Conv2d(512, 512, kernel_size=1)
        self.res5_7 = nn.Conv2d(512, 1024, kernel_size=1)
        self.res6_8 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.res7_9 = nn.Conv2d(1024, 512, kernel_size=1)
        self.res8_10 = nn.Conv2d(512, 256, kernel_size=1)
        self.res9_11 = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, X):
        # conv 1
        X = self.act(self.conv1_1(X))
        X = self.act(self.conv1_2(X))
        X = self.dout(X)
        res_1 = X
        X = self.pool1(X)

        # conv 2
        X = self.act(self.conv2_1(X))
        X = self.act(self.conv2_2(X))
        X = self.dout(X)
        res_2 = X
        X = self.pool2(X)

        # conv 3
        res_1 = tt.Resize((X.shape[2], X.shape[2]))(res_1)

        X += self.act(self.res1_3(res_1))
        X = self.act(self.conv3_1(X))
        X = self.act(self.conv3_2(X))
        X = self.dout(X)
        X = self.act(self.conv3_3(X))
        res_3 = X
        X = self.pool3(X)

        # conv 4
        res_1 = tt.Resize((X.shape[2], X.shape[2]))(res_1)
        res_2 = tt.Resize((X.shape[2], X.shape[2]))(res_2)

        X += self.act(self.res2_4(res_2))
        X = torch.cat([X, res_1], axis=1)
        X = self.act(self.conv4_1(X))
        X = self.act(self.conv4_2(X))
        X = self.dout(X)
        X = self.act(self.conv4_3(X))  # conv4_3_res
        conv4_3_res = X
        res_4 = X
        X = self.dout(X)
        X = self.pool4(X)

        # conv 5
        res_1 = tt.Resize((X.shape[2], X.shape[2]))(res_1)
        res_2 = tt.Resize((X.shape[2], X.shape[2]))(res_2)
        res_3 = tt.Resize((X.shape[2], X.shape[2]))(res_3)

        X += self.act(self.res3_5(res_3))
        X = torch.cat([X, res_1, res_2], axis=1)
        X = self.act(self.conv5_1(X))
        X = self.dout(X)
        X = self.act(self.conv5_2(X))
        X = self.act(self.conv5_3(X))
        X = self.pool5(X)
        res_5 = X

        # conv 6
        res_1 = tt.Resize((X.shape[2], X.shape[2]))(res_1)
        res_2 = tt.Resize((X.shape[2], X.shape[2]))(res_2)
        res_3 = tt.Resize((X.shape[2], X.shape[2]))(res_3)
        res_4 = tt.Resize((X.shape[2], X.shape[2]))(res_4)

        X += self.act(self.res4_6(res_4))
        X = torch.cat([X, res_1, res_2, res_3], axis=1)
        X = self.act(self.conv6(X))
        X = self.dout(X)
        res_6 = X

        # conv 7
        res_1 = tt.Resize((X.shape[2], X.shape[2]))(res_1)
        res_2 = tt.Resize((X.shape[2], X.shape[2]))(res_2)
        res_3 = tt.Resize((X.shape[2], X.shape[2]))(res_3)
        res_4 = tt.Resize((X.shape[2], X.shape[2]))(res_4)
        res_5 = tt.Resize((X.shape[2], X.shape[2]))(res_5)

        X += self.act(self.res5_7(res_5))
        X = torch.cat([X, res_1, res_2, res_3, res_4], axis=1)
        X = self.act(self.conv7(X))  # conv7_res
        X = self.dout(X)
        conv7_res = X
        res_7 = X

        # conv 8
        res_1 = tt.Resize((X.shape[2], X.shape[2]))(res_1)
        res_2 = tt.Resize((X.shape[2], X.shape[2]))(res_2)
        res_3 = tt.Resize((X.shape[2], X.shape[2]))(res_3)
        res_4 = tt.Resize((X.shape[2], X.shape[2]))(res_4)
        res_5 = tt.Resize((X.shape[2], X.shape[2]))(res_5)
        res_6 = tt.Resize((X.shape[2], X.shape[2]))(res_6)

        X += self.act(self.res6_8(res_6))
        X = torch.cat([X, res_1, res_2, res_3, res_4, res_5], axis=1)
        X = self.act(self.conv8_1(X))
        X = self.dout(X)
        X = self.act(self.conv8_2(X))  # conv8_2_res
        res_8 = X
        conv8_2_res = X

        # conv 9
        res_1 = tt.Resize((X.shape[2], X.shape[2]))(res_1)
        res_2 = tt.Resize((X.shape[2], X.shape[2]))(res_2)
        res_3 = tt.Resize((X.shape[2], X.shape[2]))(res_3)
        res_4 = tt.Resize((X.shape[2], X.shape[2]))(res_4)
        res_5 = tt.Resize((X.shape[2], X.shape[2]))(res_5)
        res_6 = tt.Resize((X.shape[2], X.shape[2]))(res_6)
        res_7 = tt.Resize((X.shape[2], X.shape[2]))(res_7)

        X += self.act(self.res7_9(res_7))
        X = torch.cat([X, res_1, res_2, res_3, res_4, res_5, res_6], axis=1)
        X = self.act(self.conv9_1(X))
        X = self.dout(X)
        X = self.act(self.conv9_2(X))  # conv9_2_res
        res_9 = X
        conv9_2_res = X

        # conv 10
        res_1 = tt.Resize((X.shape[2], X.shape[2]))(res_1)
        res_2 = tt.Resize((X.shape[2], X.shape[2]))(res_2)
        res_3 = tt.Resize((X.shape[2], X.shape[2]))(res_3)
        res_4 = tt.Resize((X.shape[2], X.shape[2]))(res_4)
        res_5 = tt.Resize((X.shape[2], X.shape[2]))(res_5)
        res_6 = tt.Resize((X.shape[2], X.shape[2]))(res_6)
        res_7 = tt.Resize((X.shape[2], X.shape[2]))(res_7)
        res_8 = tt.Resize((X.shape[2], X.shape[2]))(res_8)

        X += self.act(self.res8_10(res_8))
        X = torch.cat([X, res_1, res_2, res_3, res_4, res_5, res_6, res_7], axis=1)
        X = self.act(self.conv10_1(X))
        X = self.dout(X)
        X = self.act(self.conv10_2(X))  # conv10_2_res
        conv10_2_res = X

        # conv 11
        res_1 = tt.Resize((X.shape[2], X.shape[2]))(res_1)
        res_2 = tt.Resize((X.shape[2], X.shape[2]))(res_2)
        res_3 = tt.Resize((X.shape[2], X.shape[2]))(res_3)
        res_4 = tt.Resize((X.shape[2], X.shape[2]))(res_4)
        res_5 = tt.Resize((X.shape[2], X.shape[2]))(res_5)
        res_6 = tt.Resize((X.shape[2], X.shape[2]))(res_6)
        res_7 = tt.Resize((X.shape[2], X.shape[2]))(res_7)
        res_8 = tt.Resize((X.shape[2], X.shape[2]))(res_8)
        res_9 = tt.Resize((X.shape[2], X.shape[2]))(res_9)

        X += self.act(self.res9_11(res_9))
        X = torch.cat([X, res_1, res_2, res_3, res_4, res_5, res_6, res_7, res_8], axis=1)
        X = self.act(self.conv11_1(X))
        X = self.dout(X)
        X = self.act(self.conv11_2(X))  # conv11_2

        boxes = self.boxes_pred(conv4_3_res, conv7_res, conv8_2_res, conv9_2_res, conv10_2_res, X)
        classes = self.classes_pred(conv4_3_res, conv7_res, conv8_2_res, conv9_2_res, conv10_2_res, X)
        return boxes, classes
