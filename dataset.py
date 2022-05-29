import torch
from torchvision import transforms as tt
import os
from PIL import Image
import xml.etree.ElementTree as ET


train_path = 'C:/Users/MSI Bravo/Downloads/train_zip/train'
train_xml = ''
test_path = 'C:/Users/MSI Bravo/Downloads/test_zip/test'
test_xml = ''


train_pics_name = [i for i in sorted(os.listdir(train_path)) if i[-4:] == '.png']
train_xmls = [i for i in sorted(os.listdir(train_path)) if i[-4:] == '.xml']
test_pics_name = [i for i in sorted(os.listdir(test_path)) if i[-4:] == '.png']
test_xmls = [i for i in sorted(os.listdir(test_path)) if i[-4:] == '.xml']


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, test=False):
        self.test = test
        self.classes = ['background', 'Car']
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

        y = dict()
        y['boxes'] = list()
        y['labels'] = list()

        if not self.test:
            xml_path = os.path.join(train_xml, train_xmls[i])
        else:
            xml_path = os.path.join(test_path, test_xmls[i])

        tree = ET.parse(xml_path)
        root = tree.getroot()
        n_objs = len(root) - 5
        for i in range(n_objs):
            y['labels'].append(self.classes.index(root[5 + i][0].text))
            x_min = float(root[5 + i][4][0].text)
            y_min = float(root[5 + i][4][1].text)
            x_max = float(root[5 + i][4][2].text)
            y_max = float(root[5 + i][4][3].text)

            # scaling coordinates to 300x300 image
            x_min = x_min / width
            y_min = y_min / height
            x_max = x_max / width
            y_max = y_max / height

            y['boxes'].append([x_min, y_min, x_max, y_max])
        y['labels'] = torch.as_tensor(y['labels'])
        y['boxes'] = torch.as_tensor(y['boxes'])

        return img, y
