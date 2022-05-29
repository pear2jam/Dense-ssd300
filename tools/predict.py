import matplotlib.pyplot as plt
from matplotlib import patches

from tools.misc import *
from tools.train_functions import make_default_boxes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def simple_supression_predict(boxes, classes_pred):
    res_boxes = []
    scores = []
    labels = []
    classes = ['background', 'Car']

    pos_ind = np.where((torch.argsort(-torch.softmax(classes_pred, axis=1))[:, 0] > 0).cpu().numpy())[0]
    boxes_score = torch.argsort(-torch.max(torch.softmax(classes_pred[pos_ind], axis=1), axis=1)[0]).cpu().numpy()
    a = 0.1
    for i in pos_ind[boxes_score]:
        add = True
        for box in res_boxes:
            if IoU(box, boxes[i]) > a:
                add = False
                break
        if add:
            res_boxes.append(boxes[i].cpu().detach().numpy())
            scores.append(torch.max(torch.softmax(classes_pred, axis=1), axis=1)[0][i].cpu().detach().numpy())
            labels.append(classes[torch.argmax(classes_pred[i]).cpu().numpy()])
    return np.stack(res_boxes), scores, labels


def predict_plot(net, img, save=-1):
    net.eval()
    default = make_default_boxes()
    img = move_to(img, device)

    boxes_pred, classes_pred = net(img.view(1, 3, 300, 300))
    boxes = boxes_pred + move_to(default, device)
    classes = classes_pred[0]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    boxs, scores, labels = simple_supression_predict(boxes[0], classes)

    boxs *= 300
    ax.imshow(move_to(img, torch.device('cpu')).permute(1, 2, 0))
    for i in range(len(boxs)):
        box = boxs[i]
        w, h = box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle((box[0], box[1]), w, h, linewidth=2, facecolor='none', edgecolor='b')
        ax.add_patch(rect)
        ax.text(box[2] - 50, box[1] - 1, labels[i] + ' ' + str(scores[i]), fontsize=15)

    ax.set_title(f'{len(boxs)} boxes')

    fig.show()
    plt.show()

    if save != -1:
        fig.savefig(f'fig_{save}.png')