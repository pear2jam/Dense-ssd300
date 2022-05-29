import torch
import numpy as np

from tools.misc import IoU


def make_default_boxes():
    maps_size = {
        "conv_4": 38,
        "conv_7": 19,
        "conv_8": 10,
        "conv_9": 5,
        "conv_10": 3,
        "conv_11": 1
    }

    prior_scale = {
        "conv_4": 0.1,
        "conv_7": 0.2,
        "conv_8": 0.375,
        "conv_9": 0.55,
        "conv_10": 0.725,
        "conv_11": 0.9
    }

    aspect_ratios = {
        "conv_4": [1, 2, 1 / 2],
        "conv_7": [1, 2, 3, 1 / 2, 1 / 3],
        "conv_8": [1, 2, 3, 1 / 2, 1 / 3],
        "conv_9": [1, 2, 3, 1 / 2, 1 / 3],
        "conv_10": [1, 2, 1 / 2],
        "conv_11": [1, 2, 1 / 2]
    }

    default_boxes = []
    keys = list(maps_size.keys())

    for key, value in maps_size.items():
        for i in range(0, value):
            for j in range(0, value):
                cx = (i + 0.5) / value
                cy = (j + 0.5) / value

                for ratio in aspect_ratios[key]:
                    # using half of values for more convinient converting from
                    # center-oriented format to corners-oriented format
                    w_half = prior_scale[key] * np.sqrt(ratio) / 2  # width
                    h_half = prior_scale[key] / np.sqrt(ratio) / 2  # height
                    default_boxes.append([cx - w_half, cy - h_half, cx + w_half, cy + h_half])

                    if ratio == 1:
                        if key != "conv_11":
                            scale_1 = prior_scale[key]
                            scale_2 = prior_scale[keys[keys.index(key) + 1]]
                            scale = np.sqrt(scale_1 * scale_2)

                            default_boxes.append([cx - scale / 2, cy - scale / 2, cx + scale / 2, cy + scale / 2])
                        else:
                            default_boxes.append([cx - w_half, cy - h_half, cx + w_half, cy + h_half])

    default_boxes = torch.FloatTensor(default_boxes)
    default_boxes.retains_graph = True
    return default_boxes.clamp(0, 1)


def matching(target, boxes_pred):
    min_overlap = 0.5
    boxes = target[1]["boxes"]
    match_table = torch.IntTensor(len(boxes_pred), 2)  # assign each box their matched GT and label

    for prior in range(len(boxes_pred)):
        res_matching = np.zeros(len(boxes))  # IoUs vector of prior box with GT boxes

        for box in range(len(boxes)):
            res_matching[box] = IoU(boxes_pred[prior], target[1]["boxes"][box])

        if len(boxes) > 0:
            box_target = target[1]["labels"][np.argmax(res_matching)]
            if max(res_matching) < min_overlap:
                box_target = 0  # is bgd

            match_table[prior][0] = np.argmax(res_matching)
            match_table[prior][1] = box_target

        else:
            match_table[prior][0] = 0
            match_table[prior][1] = 0
    return match_table
