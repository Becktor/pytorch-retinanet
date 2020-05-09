import argparse
import collections
import datetime as dt
import numpy as np

import torch
import os

if os.name == 'nt':
    import ctypes

    ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from retinanet import model
from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader
from retinanet import csv_eval
from utils import *

assert torch.__version__.split('.')[0] == '1'


# print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=2)
    parser.add_argument('--noise', help='Batch size', type=bool, default=False)
    parser.add_argument('--continue_training', help='Path to previous ckp', type=str, default=None)

    parser = parser.parse_args(args)

    dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    if parser.csv_val is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                 transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(nm_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True
    if use_gpu:
        retinanet = retinanet.cuda()
    prev_epoch = 0
    boat_mAP = 0
    buoy_mAP = 0
    retinanet = torch.nn.DataParallel(retinanet).cuda()
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    checkpoint_dir = 'models/retinanet' + dt.datetime.now().strftime("%j_%H%M")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if parser.continue_training is not None:
        retinanet, optimizer, checkpoint_dict = load_ckp(parser.continue_training, retinanet, optimizer)
        checkpoint_dir = parser.continue_training
        prev_epoch = checkpoint_dict['epoch']
        boat_mAP = checkpoint_dict['boat_mAP']
        buoy_mAP = checkpoint_dict['buoy_mAP']
    writer = SummaryWriter(checkpoint_dir + "/tb_event")
    retinanet.training = True
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)
    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):
        curr_epoch = prev_epoch + epoch_num
        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss
                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))
                if iter_num % 1 == 0:
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | '
                        'Running loss: {:1.5f}'.format(
                            curr_epoch, iter_num, float(classification_loss), float(regression_loss),
                            np.mean(loss_hist)), end='\r')

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.csv_val is not None:
            print('Evaluating dataset')
            mAP, rl = csv_eval.evaluate(dataset_val, retinanet, 0.3, 0.7)

        scheduler.step(np.mean(epoch_loss))

        writer.add_scalar("train/running_loss:", np.mean(loss_hist), epoch_num)
        writer.add_scalar("val/Buoy_mAP:", rl[2][1], epoch_num)
        writer.add_scalar("val/Boat_mAP:", rl[3][1], epoch_num)

        writer.add_scalar("val/Buoy_Precision:", rl[0][2], epoch_num)
        writer.add_scalar("val/Buoy_Recall:", rl[0][1], epoch_num)

        writer.add_scalar("val/Boat_Precision:", rl[1][2], epoch_num)
        writer.add_scalar("val/Boat_Recall:", rl[1][1], epoch_num)

        checkpoint = {
            'epoch': curr_epoch + 1,
            'state_dict': retinanet.state_dict(),
            'optimizer': optimizer.state_dict(),
            'buoy_mAP': rl[2][1],
            'boat_mAP': rl[3][1]
        }

        if rl[2][1] > boat_mAP and rl[3][1] > buoy_mAP:
            boat_mAP = rl[3][1]
            buoy_mAP = rl[2][1]
            save_ckp(checkpoint, True, checkpoint_dir, curr_epoch)
        else:
            save_ckp(checkpoint, False, checkpoint_dir, curr_epoch)

        loss_file = open(checkpoint_dir + r"\loss.csv", "a+")
        loss_file.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(curr_epoch, np.mean(loss_hist),
                                                                  rl[0], rl[1], rl[2], rl[3], buoy_mAP, boat_mAP))
        loss_file.close()

    retinanet.eval()
    torch.save(retinanet, 'model_final.pt')


if __name__ == '__main__':
    main()
