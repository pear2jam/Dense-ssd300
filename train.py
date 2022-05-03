import torch

from matplotlib import pyplot as plt
from matplotlib import patches

from torchvision import transforms as tt
from PIL import Image
import xml.etree.ElementTree as ET
import os

from tools import *
from network import *
from losses import *



train_path = 'C:/Users/MSI Bravo/Downloads/train_zip/train'
test_path = 'C:/Users/MSI Bravo/Downloads/test_zip/test'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_part = 0.01
batch_size = 1
epochs = 1

sorted(os.listdir(train_path))
train_pics_name = [i for i in sorted(os.listdir(train_path)) if i[-4:] == '.jpg']
train_xmls = [i for i in sorted(os.listdir(train_path)) if i[-4:] == '.xml']
test_pics_name = [i for i in sorted(os.listdir(test_path)) if i[-4:] == '.jpg']
test_xmls = [i for i in sorted(os.listdir(test_path)) if i[-4:] == '.xml']


class FruitsDataset(torch.utils.data.Dataset):
    def __init__(self, test=False):
        self.test = test
        self.classes = ['background', 'apple', 'orange', 'banana']
        self.transforms = tt.Compose([
            tt.ToTensor(),
            tt.Resize((300, 300))
        ])

    def __len__(self):
        return len(train_pics_name)

    def __getitem__(self, i):
        if not self.test:
            img = Image.open(os.path.join(train_path, train_pics_name[i]))
        else:
            img = Image.open(os.path.join(test_path, test_pics_name[i]))

        width = img.width
        height = img.height

        print(os.path.join(train_path, train_pics_name[i]))
        img = self.transforms(img)
        img = img[:3, :, :]

        if img.shape[0] == 1:
            img_2 = torch.zeros([3, 300, 300])
            img_2[0] = img
            img_2[1] = img_2[0]
            img_2[2] = img_2[0]
            img = img_2

        y = {}
        y['boxes'] = list()
        y['labels'] = list()

        if not self.test:
            xml_path = os.path.join(train_path, train_xmls[i])
        else:
            xml_path = os.path.join(test_path, test_xmls[i])

        tree = ET.parse(xml_path)
        root = tree.getroot()

        n_objs = len(root) - 6

        for i in range(n_objs):
            y['labels'].append(self.classes.index(root[6 + i][0].text))
            x_min = int(root[6 + i][4][0].text)
            y_min = int(root[6 + i][4][1].text)
            x_max = int(root[6 + i][4][2].text)
            y_max = int(root[6 + i][4][3].text)

            # scaling coordiates to 300x300 image
            x_min = x_min / width
            y_min = y_min / height
            x_max = x_max / width
            y_max = y_max / height

            y['boxes'].append([x_min, y_min, x_max, y_max])
        y['labels'].append(0)
        y['labels'] = torch.as_tensor(y['labels'])
        # print(y['labels'])
        y['boxes'] = torch.as_tensor(y['boxes'])

        return (img, y)


classes = ['background', 'apple', 'orange', 'banana']


default = make_default_boxes()

tables = torch.zeros([int(train_part * 240), 8732, 2], dtype=torch.int32)
for i in range(int(train_part*240)):
    ex = FruitsDataset()[i]
    tables[i] = matching(ex, default)
    print(i)

exs = [FruitsDataset()[i] for i in range(int(train_part * 240))]

vgg = VGG()
al = Auxiliairy_layers()
boxes_det = Detection_boxes()
classes_det = Classes_regression(4)

vgg_opt = torch.optim.Adam(vgg.parameters(), lr=0.5e-4)
al_opt = torch.optim.Adam(al.parameters(), lr=0.2e-4)
boxes_opt = torch.optim.Adam(boxes_det.parameters(), lr=0.2e-4)
classes_opt = torch.optim.Adam(classes_det.parameters(), lr=0.2e-4)

vgg = move_to(vgg, device)
al = move_to(al, device)
boxes_det = move_to(boxes_det, device)
classes_det = move_to(classes_det, device)

default = move_to(default, device)

for i in range(epochs):
    start_batch = 0
    while start_batch + batch_size < (int(train_part * 240)):
        loss = 0
        for j in range(start_batch, start_batch + batch_size):
            ex = exs[j]
            img = ex[0]
            conv7_res, conv4_3_res = vgg(move_to(img.view(1, 3, 300, 300), device))
            conv8_2_res, conv9_2_res, conv10_2_res, conv11_2_res = al(conv7_res)

            boxes_pred = move_to(
                boxes_det(conv4_3_res, conv7_res, conv8_2_res, conv9_2_res, conv10_2_res, conv11_2_res), device)
            classes_pred = move_to(
                classes_det(conv4_3_res, conv7_res, conv8_2_res, conv9_2_res, conv10_2_res, conv11_2_res), device)

            boxes = move_to(boxes_pred[0] / 500 + default, device)
            loss += ssd_loss(tables[j], ex, boxes, classes_pred[0], 3)
        loss /= batch_size

        vgg_opt.zero_grad()
        al_opt.zero_grad()
        boxes_opt.zero_grad()
        classes_opt.zero_grad()

        loss.backward()

        vgg_opt.step()
        al_opt.step()
        boxes_opt.step()
        classes_opt.step()

        start_batch += batch_size
    print(loss)
