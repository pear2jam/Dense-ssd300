"""
Data augumentation tools
Include transform, color, combine augumentations and getting a random transformation
"""

import torch
import torchvision
import numpy as np
from tools import get_rotate_matrix


# Rotation

def rotate(image, angle):
    """
    Rotate image by input angle
    :param image: tensor with shape of (3, 300, 300)
    :param angle: angle in radians
    :return: tensor with shape of (3, 300, 300)
    """
    rot_mat = get_rotate_matrix(angle)
    rot_mat = rot_mat.view(1, 2, 3)
    grid = torch.nn.functional.affine_grid(rot_mat, [1, 3, 300, 300])
    rot_x = torch.nn.functional.grid_sample(image.view(1, 3, 300, 300), grid, padding_mode='border')

    return rot_x[0]


def rotate_boxes(boxes, angle):
    """
    Rotate bounding boxes by input angle
    :param boxes: (n_box, 4) tensor of boxes corners
    :param angle: angle in radians
    :return: (n_box, 4) tensor of rotated boxes corners
    """
    new_boxes = torch.FloatTensor(boxes.shape)
    for i in range(boxes.shape[0]):
        x_1, y_1, x_2, y_2 = boxes[i][0] - 0.5, boxes[i][1] - 0.5, boxes[i][2] - 0.5, boxes[i][3] - 0.5
        mat = get_rotate_matrix(-angle)[:, :-1]
        x_1_, y_1_ = mat @ (torch.FloatTensor([x_1, y_1])) + 0.5
        x_2_, y_2_ = mat @ (torch.FloatTensor([x_2, y_2])) + 0.5
        x_3_, y_3_ = mat @ (torch.FloatTensor([x_1, y_2])) + 0.5
        x_4_, y_4_ = mat @ (torch.FloatTensor([x_2, y_1])) + 0.5

        x_1_new = min(x_1_, x_2_, x_3_, x_4_)
        x_2_new = max(x_1_, x_2_, x_3_, x_4_)
        y_1_new = min(y_1_, y_2_, y_3_, y_4_)
        y_2_new = max(y_1_, y_2_, y_3_, y_4_)

        new_boxes[i] = torch.FloatTensor([x_1_new, y_1_new, x_2_new, y_2_new])
    return new_boxes


def rotate_sample(a, angle):
    """
    Return rotated sample by input angle
    :param a: sample of Dataset
    :param angle: angle in radians
    :return: rotated sample
    """
    return rotate(a[0], angle), dict(boxes=rotate_boxes(a[1]['boxes'], angle), labels=a[1]['labels'])


# Flip

def flip(image):
    """
    Flips input image
    :param image: tensor with shape of (3, 300, 300)
    :return: tensor with shape of (3, 300, 300)
    """
    return torch.flip(image, [2])


def flip_boxes(boxes):
    new_boxes = torch.FloatTensor(boxes.shape)
    for i in range(boxes.shape[0]):
        x_1, y_1, x_2, y_2 = 1 - boxes[i][0], boxes[i][1], 1 - boxes[i][2], boxes[i][3]
        new_boxes[i] = torch.FloatTensor([x_2, y_1, x_1, y_2])
    return new_boxes


def flip_sample(a):
    return flip(a[0]), dict(boxes=flip_boxes(a[1]['boxes']), labels=a[1]['labels'])


# Noise

def noise(image, a):
    result = image.clone()
    white = 1 - a / 2
    black = a / 2
    for i in range(300):
        for j in range(300):
            rand = np.random.random()
            if rand < black:
                result[0][i][j] = 0
                result[1][i][j] = 0
                result[2][i][j] = 0
            if rand > white:
                result[0][i][j] = 255
                result[1][i][j] = 255
                result[2][i][j] = 255
    return result


def noise_sample(a, part):
    return noise(a[0], part), dict(boxes=a[1]['boxes'], labels=a[1]['labels'])


# Shift

def shift(image, x, y):
    image_new = np.random.random([3, 300, 300])
    x = int(300 * x)
    y = int(300 * y)
    image_new[:, max(0, y):min(300, 300 + y), max(0, x):min(300, 300 + x)] = image[:, max(0, -y):max(0, 300 - y),
                                                                             max(-x, 0):max(0, 300 - x)]
    return torch.Tensor(image_new)


def shift_boxes(boxes, x, y):
    new_boxes = torch.FloatTensor(boxes.shape)
    for i in range(boxes.shape[0]):
        x_1, y_1, x_2, y_2 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        x_1 = max(0, min(300, x_1 + x))
        x_2 = max(0, min(300, x_2 + x))
        y_1 = max(0, min(300, y_1 + y))
        y_2 = max(0, min(300, y_2 + y))
        new_boxes[i] = torch.FloatTensor([x_1, y_1, x_2, y_2])
    return new_boxes


def shift_sample(a, x, y):
    return shift(a[0], x, y), dict(boxes=shift_boxes(a[1]['boxes'], x, y), labels=a[1]['labels'])


# Brightness

def bright(image, a):
    return torchvision.transforms.functional.adjust_brightness(image, a)


def bright_sample(a, brightness):
    return bright(a[0], brightness), dict(boxes=a[1]['boxes'], labels=a[1]['labels'])


# Color_Hue

def hue(image, a):
    return torchvision.transforms.functional.adjust_hue(image, a)


def hue_sample(a, hue_val):
    return hue(a[0], hue_val), dict(boxes=a[1]['boxes'], labels=a[1]['labels'])


# Combine 2 images

def combine_2(img_1, img_2, border=0.5, vertical=True):
    img = torch.FloatTensor(3, 300, 300)
    border = int(300 * border)
    if vertical:
        img[:, :, :border] = img_1[:, :, :border]
        img[:, :, border:] = img_2[:, :, border:]
    else:
        img[:, :border, :] = img_1[:, :border, :]
        img[:, border:, :] = img_2[:, border:, :]

    return img


def combine_2_boxes(boxes_1, boxes_2, border=0.5, vertical=True):
    new_boxes = torch.FloatTensor(boxes_1.shape[0] + boxes_2.shape[0], 4)
    for i in range(boxes_1.shape[0]):
        x_1, y_1, x_2, y_2 = boxes_1[i][0], boxes_1[i][1], boxes_1[i][2], boxes_1[i][3]
        if vertical:
            new_boxes[i] = torch.FloatTensor([min(border, x_1), y_1, min(border, x_2), y_2])
        else:
            new_boxes[i] = torch.FloatTensor([x_1, min(border, y_1.numpy()), x_2, min(border, y_2.numpy())])
    for i in range(boxes_2.shape[0]):
        x_1, y_1, x_2, y_2 = boxes_2[i][0], boxes_2[i][1], boxes_2[i][2], boxes_2[i][3]
        if vertical:
            new_boxes[i + boxes_1.shape[0]] = torch.FloatTensor([max(border, x_1), y_1, max(border, x_2), y_2])
        else:
            new_boxes[i + boxes_1.shape[0]] = torch.FloatTensor([x_1, max(border, y_1), x_2, max(border, y_2)])
    return new_boxes


def combine_2_sample(a_1, a_2, split, vertical=False):
    return (combine_2(a_1[0], a_2[0], split, vertical),
            dict(boxes=combine_2_boxes(a_1[1]['boxes'], a_2[1]['boxes'], split, vertical),
                 labels=torch.cat([a_1[1]['labels'], a_2[1]['labels']])))


# Combine 4 images

def combine_4(img_1, img_2, img_3, img_4, split_x, split_y):
    split_x = int(300 * split_x)
    split_y = int(300 * split_y)
    img = torch.FloatTensor(3, 300, 300)
    img[:, :split_x, :split_y] = torchvision.transforms.Resize((split_x, split_y))(img_1)
    img[:, split_x:, :split_y] = torchvision.transforms.Resize((300 - split_x, split_y))(img_2)
    img[:, :split_x, split_y:] = torchvision.transforms.Resize((split_x, 300 - split_y))(img_3)
    img[:, split_x:, split_y:] = torchvision.transforms.Resize((300 - split_x, 300 - split_y))(img_4)

    return img


def combine_4_boxes(boxes_1, boxes_2, boxes_3, boxes_4, split_y, split_x):
    new_boxes = torch.FloatTensor(boxes_1.shape[0] + boxes_2.shape[0] + boxes_3.shape[0] + boxes_4.shape[0], 4)
    for i in range(boxes_1.shape[0]):
        x_1, y_1, x_2, y_2 = boxes_1[i][0] * split_x, boxes_1[i][1] * split_y, boxes_1[i][2] * split_x, boxes_1[i][
            3] * split_y
        new_boxes[i] = torch.FloatTensor([x_1, y_1, x_2, y_2])

    for i in range(boxes_2.shape[0]):
        x_1, y_1, x_2, y_2 = boxes_2[i][0] * split_x, boxes_2[i][1] * (1 - split_y) + split_y, boxes_2[i][2] * split_x, \
                             boxes_2[i][3] * (1 - split_y) + split_y
        new_boxes[i + boxes_1.shape[0]] = torch.FloatTensor([x_1, y_1, x_2, y_2])

    for i in range(boxes_3.shape[0]):
        x_1, y_1, x_2, y_2 = boxes_3[i][0] * (1 - split_x) + split_x, boxes_3[i][1] * split_y, boxes_3[i][2] * (
                1 - split_x) + split_x, boxes_3[i][3] * split_y
        new_boxes[i + boxes_1.shape[0] + boxes_2.shape[0]] = torch.FloatTensor([x_1, y_1, x_2, y_2])

    for i in range(boxes_4.shape[0]):
        x_1, y_1, x_2, y_2 = boxes_4[i][0] * (1 - split_x) + split_x, boxes_4[i][1] * (1 - split_y) + split_y, \
                             boxes_4[i][2] * (1 - split_x) + split_x, boxes_4[i][3] * (1 - split_y) + split_y
        new_boxes[i + boxes_1.shape[0] + boxes_2.shape[0] + boxes_3.shape[0]] = \
            torch.FloatTensor([x_1, y_1, x_2, y_2])
    return new_boxes


def combine_4_sample(a_1, a_2, a_3, a_4, x=0.5, y=0.5):
    return (combine_4(a_1[0], a_2[0], a_3[0], a_4[0], x, y), dict(boxes= \
                                                                      combine_4_boxes(a_1[1]['boxes'], a_2[1]['boxes'],
                                                                                      a_3[1]['boxes'], a_4[1]['boxes'],
                                                                                      x, y), labels=torch.cat(
        [a_1[1]['labels'], a_2[1]['labels'], a_3[1]['labels'], a_4[1]['labels']])))


# Get random transformation

def random_transform_image(data):
    def random_transform(image):
        image = rotate_sample(image, np.random.random() * 0.5 - 0.25)  # from -0.1 to 0.1

        if np.random.random() > 0.4:
            image = flip_sample(image)

        if np.random.random() > 0.5:
            image = noise_sample(image, np.random.random() / 8)  # from 0 to 1/8

        if np.random.random() > 0.5:
            image = shift_sample(image, np.random.random() * 0.6 - 0.3,
                                 np.random.random() * 0.6 - 0.3)  # both from -0.3 to -0.3

        if np.random.random() > 0.2:
            image = bright_sample(image, np.random.random() * 1.4 + 0.3)  # from 0.3 to 1.7
            image = hue_sample(image, np.random.random() - 0.5)
        return image

    type_transform = np.random.random()

    if 0.6 < type_transform < 0.9:
        img_1 = data[np.random.randint(0, len(data))]
        img_2 = data[np.random.randint(0, len(data))]
        if np.random.random() > 0.7:
            img_1 = random_transform(img_1)
        if np.random.random() > 0.7:
            img_2 = random_transform(img_2)
        return combine_2_sample(img_1, img_2, np.random.random() * 0.2 + 0.4)
    if type_transform > 0.9:
        img_1 = data[np.random.randint(0, len(data))]
        img_2 = data[np.random.randint(0, len(data))]
        img_3 = data[np.random.randint(0, len(data))]
        img_4 = data[np.random.randint(0, len(data))]
        if np.random.random() > 0.75:
            img_1 = random_transform(img_1)
        if np.random.random() > 0.75:
            img_2 = random_transform(img_2)
        if np.random.random() > 0.75:
            img_3 = random_transform(img_3)
        if np.random.random() > 0.75:
            img_4 = random_transform(img_4)
        return combine_4_sample(img_1, img_2, img_3, img_4, np.random.random() * 0.2 + 0.4,
                                np.random.random() * 0.2 + 0.4)
    else:
        return random_transform(data[np.random.randint(0, len(data))])
