import argparse
import csv

import time

from tensorflow.keras.optimizers import Adam

from tensorboardX import SummaryWriter

import custom_transforms
from convert import *
from logger import AverageMeter
from models import PSNet as PSNet
from sequence_folders import SequenceFolder
from utils import tensor2array, save_path_formatter



parser = argparse.ArgumentParser(description='DeepSFM depth subnet train script',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=4e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--geo', '--geo-cost', default=True, type=bool,
                    metavar='GC', help='whether add geometry cost')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained-dps', dest='pretrained_dps',
                    default='', metavar='PATH',
                    help='path to pre-trained dispnet model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('--ttype', default='train.txt', type=str, help='Text file indicates input data')
parser.add_argument('--ttype2', default='val.txt', type=str, help='Text file indicates input data')
parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=100)
parser.add_argument('--nlabel', type=int ,default=64, help='number of label')
parser.add_argument('--mindepth', type=float ,default=0.5, help='minimum depth')
parser.add_argument('--pose_init', default='demon', help='path to init pose')
parser.add_argument('--depth_init', default='demon', help='path to init depth')

n_iter = 0


def main():
    global n_iter
    args = parser.parse_args()
    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints'/save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        ttype=args.ttype,
        add_geo=args.geo,
        depth_source=args.depth_init,
        pose_source='%s_poses.txt'%args.pose_init if args.pose_init else 'poses.txt',
        scale=False
    )
    val_set = SequenceFolder(
        args.data,
        transform=valid_transform,
        seed=args.seed,
        ttype=args.ttype2,
        add_geo=args.geo,
        depth_source=args.depth_init,
        pose_source='%s_poses.txt' % args.pose_init if args.pose_init else 'poses.txt',
        scale=False
    )

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))

    if args.epoch_size == 0:
        args.epoch_size = len(train_set)

    # create model
    print("=> creating model")
    depth_net = PSNet(args.nlabel, args.mindepth, add_geo_cost=args.geo, depth_augment=False)

    print('=> setting adam solver')

    optimizer = Adam(args.lr, beta_1=args.momentum, beta_2=args.beta)

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss'])


    for epoch in range(args.epochs):
        learning_rate = args.lr * (0.8 ** (epoch // 10))
        optimizer.learning_rate.assign(learning_rate)
        # train for one epoch
        
        train_loss = train(args, train_loader, depth_net, optimizer, args.epoch_size, training_writer)

        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': depth_net.module.state_dict()
            },
            epoch)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss])


if __name__ == "__main__":
    main()
