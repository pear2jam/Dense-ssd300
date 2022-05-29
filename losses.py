import numpy as np
import torch.nn as nn
import torch

from tools.misc import move_to


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loc_loss(table, boxes, target):
    pos_ind = np.where(table[:,1] > 0)[0]
    loss = 0
    if pos_ind.shape[0] == 0:
        return loss
    for pos_box in pos_ind:
        loss += nn.SmoothL1Loss(beta=1/300)(boxes[pos_box], move_to(target[1]["boxes"][table[pos_box][0]], device))
    return loss / pos_ind.size


def conf_loss(table, classes_pred, hn=3):
    def hard_negatives(table, classes_pred):
        neg_ind = np.where(table[:, 1] == 0)[0]
        n_pos = (table[:, 1] > 0).sum()
        if n_pos == 0:
            n_pos = 6
        n_neg = min((table[:, 1] == 0).sum(), hn * n_pos)
        false_neg_pred = torch.softmax(classes_pred[neg_ind], axis=1)[:, 0]
        hard_neg_slice = np.argsort(false_neg_pred.detach().cpu())[:n_neg]
        return neg_ind[hard_neg_slice]

    pos_ind = np.where(table[:, 1] > 0)[0]
    loss = 0

    for pos_box in pos_ind:
        a = classes_pred[pos_box].reshape(1, -1)
        b = move_to(table[pos_box][1].reshape(1).long(), device)
        loss += torch.nn.CrossEntropyLoss()(a, b)
    for hard_neg in hard_negatives(table, classes_pred):
        a = classes_pred[hard_neg].reshape(1, -1)
        loss += torch.nn.CrossEntropyLoss()(a, move_to(torch.LongTensor([0]), device))
    if len(pos_ind) == 0:
        loss /= 6
    else:
        loss /= len(pos_ind)
    return loss


def ssd_loss(table, target, boxes_pred, classes_pred, a=1, hn=3):
    loc = loc_loss(table, boxes_pred, target) * a
    conf = conf_loss(table, classes_pred, hn)
    return loc + conf
