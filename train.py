from network import *
from tools import *
from dataset import MyDataset
from losses import *
from dataset import MyDataset
from augmentation import *


# Data
train_aug = 1
test_aug = 1
test = True

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 1
batch_size = 1
lr = 3e-4

# Model
dropout_rate = 0


classes = ['background', 'Car']

train_data = []
test_data = []

train_dataset = MyDataset()
if test:
    test_dataset = MyDataset(test=True)

for i in range(len(train_data)):
    train_data.append(train_data[i])

if test:
    for i in range(len(test_data)):
        test_data.append(test_data[i])

train_data_len = len(train_data)
test_data_len = len(test_data)

print("Before Augmentation:")
print(f'Train: {train_data_len}, Test:{test_data_len}')

# Augmentation

for i in range(train_aug):
    train_data.append(random_transform_image(train_data[:train_data_len]))

if test:
    for i in range(test_aug):
        test_data.append(random_transform_image(test_data_len[:test_data_len]))

print("After Augmentation:")
print(f'Train: {len(train_data)}, Test:{len(test_data)}')

#  Calculating tables
default = make_default_boxes()

train_tables = torch.zeros([len(train_data), 8732, 2], dtype=torch.int32)
if test:
    test_tables = torch.zeros([len(test_data), 8732, 2], dtype=torch.int32)

print('Train:', end='')
for i in range(len(train_data)):
    ex = train_data[i]
    train_tables[i] = matching(ex, default)
    if i % 20 == 0:
        print()
    print(i, end=' ')

print()
if test:
    print('Test:', end='')
    for i in range(len(test_data)):
        ex = test_data[i]
        test_tables[i] = matching(ex, default)
        if i % 20 == 0:
            print()
        print(i, end=' ')


ssd300 = SSD300(len(train_dataset.classes), dropout_rate)
ssd_opt = torch.optim.Adam(ssd300.parameters(), lr=lr)

ssd300 = move_to(ssd300, device)

for epoch in range(epochs):
    ssd300.train()
    train_loss = 0
    start_batch = 0
    while start_batch + batch_size <= (len(train_data)):
        loss = 0
        for j in range(start_batch, start_batch + batch_size):
            ex = train_data[j]
            img = ex[0]
            boxes_pred, classes_pred = ssd300(move_to(img.view(1, 3, 300, 300), device))
            boxes = move_to(boxes_pred[0] + default, device)
            loss += ssd_loss(train_tables[j], ex, boxes, classes_pred[0])
        loss /= batch_size

        ssd_opt.zero_grad()

        loss.backward()

        ssd_opt.step()

        train_loss += loss.detach().cpu().numpy()
        del loss, boxes_pred, classes_pred
        torch.cuda.empty_cache()

        start_batch += batch_size
    ssd300.eval()

    if test:
        test_loss = 0
        for j in range(test_data_len):
            test_loss = 0
            ex = test_data[j]
            img = ex[0]
            boxes_pred, classes_pred = ssd300(move_to(img.view(1, 3, 300, 300), device))
            boxes = move_to(boxes_pred[0] + default, device)
            test_loss += ssd_loss(test_tables[j], ex, boxes, classes_pred[0], 3)
        test_loss /= len(test_data)
        test_loss = test_loss.detach().cpu().numpy()

        del boxes_pred, classes_pred
        torch.cuda.empty_cache()

    print(f'{i}: ')
    print('Train:', train_loss / (len(train_data) // batch_size))
    if test:
        print('Test', test_loss)
