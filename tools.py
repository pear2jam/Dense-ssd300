import numpy as np
import torch


def IoU(a, b):
    interception_x = min(a[2], b[2]) - max(a[0], b[0])
    interception_y = min(a[3], b[3]) - max(a[1], b[1])

    interception_x = max(0, interception_x)
    interception_y = max(0, interception_y)

    interception = interception_x * interception_y

    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - interception
    return interception / union


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
    min_overlap = 0.55
    boxes = target[1]["boxes"]
    match_table = torch.IntTensor(len(boxes_pred), 2) #assign each box their matched GT and label

    for prior in range(len(boxes_pred)):
        res_matching = np.zeros(len(boxes)) # IoUs vector of prior box with GT boxes

        for box in range(len(boxes)):
            res_matching[box] = IoU(boxes_pred[prior], target[1]["boxes"][box])

        box_target = target[1]["labels"][np.argmax(res_matching)]
        if max(res_matching) < min_overlap:
            box_target = 0  # is bgd

        match_table[prior][0] = np.argmax(res_matching)
        match_table[prior][1] = box_target

    return match_table


def simple_supression_predict(boxes, classes_pred):
    print(boxes.shape)
    res_boxes = []
    scores = []
    labels = []
    classes = ['background', 'apple', 'orange', 'banana']
    boxes_score = torch.argsort(-torch.max(torch.softmax(classes_pred, axis=1), axis=1)[0])
    a = 0.1
    for i in boxes_score:
        if torch.argmax(classes_pred[i]).detach().numpy() == 0:
            continue
        if torch.softmax(classes_pred[i], axis=0).max() < 0.1:
            continue
        add = True
        for box in res_boxes:
            if IoU(box, boxes[i]) > a:
                add = False
                break
        if add:
            res_boxes.append(boxes[i])
            scores.append(torch.max(torch.softmax(classes_pred, axis=1), axis=1)[0][i].detach().numpy())
            labels.append(classes[torch.argmax(classes_pred[i]).detach().numpy()])
    return (np.stack(res_boxes), scores, labels)


def move_to(data, device):
    if isinstance(data, (list, tuple)):
        return [move_to(x, device) for x in data]
    return data.to(device, non_blocking=True)