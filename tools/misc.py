import numpy as np
import torch


def IoU(a, b):
    # x_min, y_min, x_max, y_max

    interception_x = min(a[2], b[2]) - max(a[0], b[0])
    interception_y = min(a[3], b[3]) - max(a[1], b[1])

    interception_x = max(0, interception_x)
    interception_y = max(0, interception_y)

    interception = interception_x * interception_y

    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - interception
    return interception / union


def get_rotate_matrix(theta):  # in radians
    theta = torch.FloatTensor([theta])
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def move_to(data, device):
    if isinstance(data, (list, tuple)):
        return [move_to(x, device) for x in data]
    return data.to(device, non_blocking=True)
