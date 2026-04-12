import copy
import PIL.Image as PILImage
import os

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), \
            f'predict {inputs.size()} & target {target.size()} shape do not match'
        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i]) * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256],
                       test_save_path=None, case=None, z_spacing=1):
    image = image.squeeze(0).cpu().detach().numpy()   # [C,H,W]
    label = label.squeeze(0).cpu().detach().numpy()   # [H,W]
    _, x, y = image.shape

    if (x, y) != (patch_size[0], patch_size[1]):
        image = zoom(image, (1, patch_size[0] / x, patch_size[1] / y), order=3)

    inp = torch.from_numpy(image).unsqueeze(0).float().cuda()  # [1,C,H,W]
    net.eval()
    with torch.no_grad():
        logits = net(inp)
        out = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()  # [H,W]

        if (x, y) != (patch_size[0], patch_size[1]):
            prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            prediction = out

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None and case is not None:
        os.makedirs(test_save_path, exist_ok=True)

        a1 = copy.deepcopy(prediction)
        a2 = copy.deepcopy(prediction)
        a3 = copy.deepcopy(prediction)

        a1[a1 == 1] = 255
        a1[a1 == 2] = 0
        a1[a1 == 3] = 255
        a1[a1 == 4] = 20

        a2[a2 == 1] = 255
        a2[a2 == 2] = 255
        a2[a2 == 3] = 0
        a2[a2 == 4] = 10

        a3[a3 == 1] = 255
        a3[a3 == 2] = 77
        a3[a3 == 3] = 0
        a3[a3 == 4] = 120

        a1 = PILImage.fromarray(np.uint8(a1)).convert('L')
        a2 = PILImage.fromarray(np.uint8(a2)).convert('L')
        a3 = PILImage.fromarray(np.uint8(a3)).convert('L')
        color_pred = PILImage.merge('RGB', [a1, a2, a3])
        color_pred.save(os.path.join(test_save_path, f'{case}.png'))

    return metric_list
