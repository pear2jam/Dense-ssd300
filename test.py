from network import *
from tools import *
from dataset import FruitsDataset

from matplotlib import pyplot as plt
from matplotlib import patches
import os

vgg = VGG()
al = Auxiliairy_layers()
boxes_det = Detection_boxes()
classes_det = Classes_regression(4)

vgg.load_state_dict(torch.load('./models/vgg.pth'))
al.load_state_dict(torch.load('./models/al.pth'))
boxes_det.load_state_dict(torch.load('./models/boxes_det.pth'))
classes_det.load_state_dict(torch.load('./models/classes_det.pth'))


i = 2

default = make_default_boxes()
ex = FruitsDataset()[i]
conv7_res, conv4_3_res = move_to(vgg, torch.device('cpu'))(FruitsDataset()[i][0].view(1, 3, 300, 300))
conv8_2_res, conv9_2_res, conv10_2_res, conv11_2_res = move_to(al, torch.device('cpu'))(conv7_res)

boxes_pred = move_to(boxes_det, torch.device('cpu'))(conv4_3_res, conv7_res, conv8_2_res, conv9_2_res, conv10_2_res, conv11_2_res)[0]
classes_pred = move_to(classes_det, torch.device('cpu'))(conv4_3_res, conv7_res, conv8_2_res, conv9_2_res, conv10_2_res, conv11_2_res)

boxes = move_to(boxes_pred/500, torch.device('cpu'))[0] + move_to(default, torch.device('cpu'))
boxes = boxes.detach().numpy()
classes = classes_pred[0]


fig, ax = plt.subplots(1, 1, figsize=(8, 8))

boxs, scores, labels = simple_supression_predict(boxes, classes)


boxs *= 300
ax.imshow(ex[0].permute(1, 2, 0))
for i in range(len(boxs)):
    box = boxs[i]
    w, h = box[2]-box[0], box[3]-box[1]
    rect = patches.Rectangle((box[0], box[1]), w, h, linewidth=2, facecolor='none', edgecolor='b')
    ax.add_patch(rect)
    ax.text(box[2]-50, box[1]-1, labels[i] + ' ' + str(scores[i]), fontsize=15)


ax.set_title(f'{len(boxs)} boxes')


fig.show()
plt.show()
