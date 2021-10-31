# Code for MedT


import torch
import lib
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import torch
import torch.nn.init as init
from utils import JointTransform2D, ImageToImage2D, Image2D
from metrics import classwise_iou, LogNLLLoss, classwise_f1, FocalLoss, BCEFocalLoss, Acc, Weighted_Focal_Dice_loss, \
    BCEDiceLoss
from utils import chk_mkdir, Logger, MetricList
import cv2
from functools import partial
from random import randint
import timeit
import tensorboard
import tensorboardX
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=2, type=int,
                    metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.95, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--train_dataset', default='CRACK500/train', type=str)
parser.add_argument('--val_dataset', default='CRACK500/val', type=str)
parser.add_argument('--save_freq', type=int, default=4)

parser.add_argument('--modelname', default='MedT', type=str,
                    help='type of model')
parser.add_argument('--cuda', default="on", type=str,
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='on', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='None', type=str,
                    help='load a pretrained model')
parser.add_argument('--save', default='default', type=str,
                    help='save the model')
parser.add_argument('--direc', default='result-fl-crack500', type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int, default=256)
parser.add_argument('--imgsize', type=int, default=256)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--gray', default='no', type=str)
parser.add_argument('--num_classes', default='2', type=int)
# CUDA_VISIBLE_DEVICES = 2
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
args = parser.parse_args()
gray_ = args.gray
aug = args.aug
direc = args.direc
modelname = args.modelname
imgsize = args.imgsize
torch.backends.cudnn.enabled = False
from datetime import datetime

dir = os.path.join('tensorboard-fl-crack500', datetime.now().strftime("%Y%m%d%H%M%S"))
writer = SummaryWriter(log_dir=os.path.join('runs', dir))
if not os.path.isdir(direc):
    os.mkdir(direc)
with open(os.path.join(direc, 'args.txt'), 'a') as file:
    file.write(args.__str__())
file.close()
if gray_ == "yes":
    from utils_gray import JointTransform2D, ImageToImage2D, Image2D

    imgchant = 1
else:
    from utils import JointTransform2D, ImageToImage2D, Image2D

    imgchant = 3

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
tf_test = JointTransform2D(crop=None, p_flip=0, color_jitter_params=None, long_mask=True)
train_dataset = ImageToImage2D(args.train_dataset, tf_val)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
test_dataset = ImageToImage2D(args.val_dataset, tf_test)
predict_dataset = Image2D(args.val_dataset)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

device = torch.device("cuda")

if modelname == "axialunet":
    model = lib.models.axialunet(img_size=imgsize, imgchan=imgchant)
elif modelname == "MedT":
    model = lib.models.axialnet.MedT(img_size=imgsize, imgchan=imgchant)
elif modelname == "gatedaxialunet":
    model = lib.models.axialnet.gated(img_size=imgsize, imgchan=imgchant)
elif modelname == "logo":
    model = lib.models.axialnet.logo(img_size=imgsize, imgchan=imgchant)
elif modelname == "gated_axial_unet":
    model = lib.models.model_codes.mix_net_gated_d()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model, device_ids=[0]).cuda()

model.to(device)

# criterion = LogNLLLoss()
criterion = Weighted_Focal_Dice_loss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,
                             weight_decay=1e-5)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.set_deterministic(True)
# random.seed(seed)


i = 0
best_model = 0.
best_epoch = 0
for epoch in range(args.epochs):

    epoch_running_loss = 0

    for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):
        X_batch = Variable(X_batch.to(device='cuda'))

        y_batch = Variable(y_batch.to(device='cuda'))

        # ===================forward=====================
        #output,side2,side1,side = model(X_batch)
        output = model(X_batch)
        # tmp2 = y_batch.detach().cpu().numpy()

        # tmp2[tmp2>0] = 1
        # tmp2[tmp2<=0] = 0
        # tmp2 = tmp2.astype(int)

        y_batch[y_batch > 0] = 1
        y_batch[y_batch <= 0] = 0
        loss = criterion(output, y_batch)
      #  loss2 = criterion(side2, y_batch)
      #  loss1 = criterion(side1, y_batch)
    # loss0 = criterion(side, y_batch)
      #  total_loss = loss + loss2*0.5 + loss1*0.6 + loss0*0.7

        print('epoch_batch_idx [{}/{}], loss:{:.4f}'
              .format(epoch, batch_idx, loss.item()))
        '''
        print('epoch_batch_idx [{}/{}], loss2:{:.4f}'
              .format(epoch, batch_idx, loss2.item()))
        print('epoch_batch_idx [{}/{}], loss1:{:.4f}'
              .format(epoch, batch_idx, loss1.item()))
        print('epoch_batch_idx [{}/{}], loss0:{:.4f}'
              .format(epoch, batch_idx, loss0.item()))
'''
        # ===================backward====================
        optimizer.zero_grad()
        #total_loss.backward()
        loss.backward()
        optimizer.step()
        epoch_running_loss += loss.item()

    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch, args.epochs, epoch_running_loss / (batch_idx + 1)))
    writer.add_scalar('train_loss_epoch', epoch_running_loss / (batch_idx + 1), (epoch + 1) * (batch_idx + 1))

    if epoch >= 10:
        for param in model.parameters():
            param.requires_grad = True
    if (epoch % args.save_freq) == 0:

        total_iou = 0.
        total_f1 = 0.
        total_acc = 0.

        for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):

            if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
            else:
                image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

            X_batch = Variable(X_batch.to(device='cuda'))
            y_batch = Variable(y_batch.to(device='cuda'))
            # start = timeit.default_timer()
            x = X_batch.detach().cpu().numpy()
            #y_out,side2,side1,side = model(X_batch)
            y_out= model(X_batch)

            y_batch[y_batch > 0] = 1
            y_batch[y_batch <= 0] = 0

            # stop = timeit.default_timer()
            # print('Time: ', stop - start)
            tmp2 = y_batch.detach().cpu().numpy()  # label
            tmp = y_out.detach().cpu().numpy()  # pred


            tmp[tmp >= 0.5] = 1
            tmp[tmp < 0.5] = 0
            tmp2[tmp2 > 0] = 1
            tmp2[tmp2 <= 0] = 0
            x = x * 255

            tmp2 = tmp2.astype(int)
            tmp = tmp.astype(int)

            iou = np.array(classwise_iou(y_out.detach().cpu(), tmp2))
            f1 = np.array(classwise_f1(y_out.detach().cpu(), tmp2))
            acc = np.array(Acc(y_out.detach().cpu(), tmp2))

            total_iou += iou
            total_f1 += f1
            total_acc += acc

            print("epoch_" + str(epoch) + "  batch_idx_" + str(batch_idx) + "  iou: " + str(iou) + " f1: " + str(
                f1) + " acc: " + str(acc) + "\n")

            epsilon = 1e-20

            yHaT = tmp
            yval = tmp2

            del X_batch, y_batch, tmp, tmp2, y_out

            yHaT[yHaT == 1] = 255
            yval[yval == 1] = 255

            predict = np.expand_dims(yHaT[:, 1, :, :], axis=1).repeat(3, axis=1)
            ground = np.expand_dims(yval, axis=1).repeat(3, axis=1)

            fulldir = direc + "/{}/".format(epoch)
            if not os.path.isdir(fulldir):
                os.makedirs(fulldir)

            if np.sum(yval[0, :, :]) == 0:
                continue
            split = np.full(shape=(1, 256), fill_value=255)
            image = np.concatenate((yHaT[0, 1, :, :], split))
            image = np.concatenate((image, yval[0, :, :]))
            cv2.imwrite(fulldir + image_filename, image)

            cv2.imwrite(fulldir + 'original_' + image_filename, x[0,:,:,:])

          #  writer.add_images('val_predits', np.expand_dims(image,0), (epoch + 1) * args.batch_size * batch_idx)

        print('epoch [{}/{}], iou:{:.4f}'
              .format(epoch, args.epochs, total_iou / (batch_idx + 1)))
        writer.add_scalar('val_iou_epoch', total_iou / (batch_idx + 1), (epoch + 1))

        print('epoch [{}/{}], f1:{:.4f}'
              .format(epoch, args.epochs, total_f1 / (batch_idx + 1)))
        writer.add_scalar('val_f1_epoch', total_f1 / (batch_idx + 1), (epoch + 1))

        print('epoch [{}/{}], acc:{:.4f}'
              .format(epoch, args.epochs, total_acc / (batch_idx + 1)))
        writer.add_scalar('val_acc_epoch', total_acc / (batch_idx + 1), (epoch + 1))

        fulldir = direc + "/{}/".format(epoch)
        torch.save(model.state_dict(), fulldir + args.modelname + ".pth")
        if (total_f1 / (batch_idx + 1)) >= best_model:
            best_model = total_f1 / (batch_idx + 1)
            best_epoch = epoch
            torch.save(model.state_dict(), direc + str(best_epoch) + str(best_model) + "best_model.pth")

