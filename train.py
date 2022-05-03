from tools import *
from network import *
from losses import *
from dataset import FruitsDataset


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_part = 0.01
batch_size = 1
epochs = 1

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


torch.save(vgg.state_dict(), './models/vgg.pth')
torch.save(al.state_dict(), './models/al.pth')
torch.save(boxes_det.state_dict(), './models/boxes_det.pth')
torch.save(classes_det.state_dict(), './models/classes_det.pth')
