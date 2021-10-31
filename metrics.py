import torch
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _WeightedLoss
import os
from os.path import join
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as T
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import math
EPSILON = 1e-20
class LogNLLLoss(_WeightedLoss):
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None,
                 ignore_index=255):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, y_input, y_target):
        # y_input = torch.log(y_input + EPSILON)
        return cross_entropy(y_input, y_target, weight=self.weight,
                             ignore_index=self.ignore_index)



class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, inputs_, targets):
        gpu_targets = targets.cuda()
        inputs_ = F.sigmoid(inputs_)
        inputs = inputs_[:, 1, :, :]
        alpha_factor = torch.ones(gpu_targets.shape).cuda() * 0.25
        alpha_factor = torch.where(torch.eq(gpu_targets, 1), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(gpu_targets, 1), 1. - inputs, inputs)
        focal_weight = alpha_factor * torch.pow(focal_weight, 2)
        targets = targets.type(torch.FloatTensor)
        inputs = inputs.cuda()
        targets = targets.cuda()
        bce = F.binary_cross_entropy(inputs, targets)
        focal_weight = focal_weight.cuda()
        cls_loss = focal_weight * bce
        return cls_loss.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
       # bce = F.binary_cross_entropy(input[:, 1, :, :], target.float())
        smooth = 1e-5
        input = torch.argmax(input, dim=1)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice


class BCEFocalLoss(nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)[:, 1, :, :]
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class Weighted_Focal_Dice_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        f_loss = BCEFocalLoss()
        d_loss = BCEDiceLoss()
        f_score = f_loss(inputs, targets)
        d_score = d_loss(inputs, targets)
        total_loss = f_score * 0.4 + d_score * 0.6
        return total_loss


def Acc(output, gt): # recall
    epsilon = 1e-20
    output = torch.argmax(output, dim=1)
    t = (output * gt).sum()
    all = gt.sum()
    return (t + epsilon) / (all + epsilon)


def classwise_iou(output, gt):  #
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """
    output = torch.argmax(output, dim=1)
    # gt = np.array(gt)
    # dims = (0, *range(2, len(output.shape)))
    # gt = torch.zeros_like(output).scatter_(1, gt[:,None ,:], 1)
    intersection = output * gt
    union = output | gt
    # print(intersection.sum(axis=dims).shape)
    classwise_iou = (intersection.sum() + EPSILON) / (union.sum() + EPSILON)
    return classwise_iou


def classwise_f1(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """
    epsilon = 1e-9
    n_classes = output.shape[1]

    output = torch.argmax(output, dim=1)
    # print(output.shape)
    true_positives = ((output)*(gt)).sum()
    # print(true_positives)
    selected = output.sum()
    relevant = gt.sum()

    precision = (true_positives + epsilon)/(selected + epsilon)
    recall = (true_positives + epsilon) / (relevant + epsilon)
    classwise_f1 = 2 * (precision * recall) / (precision + recall)
    if math.isnan(classwise_f1):
        classwise_f1 = 0.

    return classwise_f1



def make_weighted_metric(classwise_metric):
    """
    Args:
        classwise_metric: classwise metric like classwise_IOU or classwise_F1
    """

    def weighted_metric(output, gt, weights=None):

        # dimensions to sum over
        dims = (0, *range(2, len(output.shape)))

        # default weights
        if weights == None:
            weights = torch.ones(output.shape[1]) / output.shape[1]
        else:
            # creating tensor if needed
            if len(weights) != output.shape[1]:
                raise ValueError("The number of weights must match with the number of classes")
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights)
            # normalizing weights
            weights /= torch.sum(weights)

        classwise_scores = classwise_metric(output, gt).cpu()

        return classwise_scores

    return weighted_metric


if __name__ == '__main__':
    gt_dir = 'dc-test-label'
    pred_dir = 'results-test-dc-logo-2'
    png_name_list = os.listdir(gt_dir)
    to_tensor = T.ToTensor()
    i = 0
    iou = 0.
    f1 = 0.
    writer = SummaryWriter(log_dir='runs/' + pred_dir)
    for file_name in png_name_list:
        mask = np.array((Image.open(os.path.join(gt_dir, file_name)).convert('L')))

        output = np.array(Image.open(os.path.join(pred_dir, file_name[:-4] + '.jpg')).convert('L'))

        mask[mask <= 127] = 0
        mask[mask > 127] = 1

        output[output <= 127] = 0
        output[output > 127] = 1

        mask = torch.unsqueeze(to_tensor(mask), 0)
        output = to_tensor(output)
        output = torch.unsqueeze(output, 0)

        gt = mask

        gt[gt == 1] = 255
        pred = output
        pred[pred == 1] = 255
        print(pred.shape)
        print(gt.shape)

        iou += np.array(classwise_iou(output, mask))
        f1 += np.array(classwise_f1(output, mask))

        iou_1 = np.squeeze(np.array(classwise_iou(output, mask)))
        f1_1 = np.array(classwise_f1(output, mask))

        writer.add_scalar('iou', iou_1, i)
        writer.add_scalar('f1', f1_1, i)
        writer.add_images('test_gt', gt, i)
        writer.add_images('test_pred', pred, i)

        i += 1

    print(i)
    print(iou / i)
    print(f1 / i)
