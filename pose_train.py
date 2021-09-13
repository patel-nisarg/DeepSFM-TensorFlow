import argparse
import csv
from operator import pos
from random import seed

import time
from cv2 import transform
import tensorflow as tf
from tensorboardX import SummaryWriter
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Softmax
from tensorflow.keras.losses import MeanAbsoluteError

import custom_transforms
from convert import *
from logger import AverageMeter
from models import PoseNet
from pose_sequence_folders import SequenceFolder
from utils import save_path_formatter

parser = argparse.ArgumentParser(description='DeepSFM pose subnet train script',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=10, type=int, metavar='N',  # 10
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=1, type=int,  # 6
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--geo', '--geo-cost', default=True, type=bool,
                    metavar='GC', help='whether add geometry cost')
parser.add_argument('--noise', '--pose-noise', default=False, type=bool,
                    metavar='PN', help='whether add pose noise')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=1, type=int,
                    metavar='N', help='print frequency')

parser.add_argument('--pretrained-dps', dest='pretrained_dps',
                    default='',
                    metavar='PATH',
                    help='path to pre-trained model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true',
                    help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('--ttype', default='train.txt', type=str, help='Text file indicates input data')
parser.add_argument('-f', '--training-output-freq', type=int,
                    help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=100)
parser.add_argument('--nlabel', type=int, default=10, help='number of label')
parser.add_argument('--std_tr', type=float, default=0.27, help='translation')
parser.add_argument('--std_rot', type=float, default=0.12, help='rotation')
parser.add_argument('--pose_init', default='demon', help='path to init pose')
parser.add_argument('--depth_init', default='demon', help='path to init depth')


n_iter = 0

def main():
    global n_iter
    args = parser.parse_args()
    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints_pose6' / save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()


    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path / 'valid' / str(i)))

    # Data transformation functions
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([custom_transforms.RandomScaleCrop(),
                                                custom_transforms.ArrayToTensor(),
                                                normalize])                                
    
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        batch_size = args.batch_size,
        seed=args.seed,
        ttype=args.ttype,
        add_geo=args.geo,
        depth_source=args.depth_init,
        gt_source='g',
        std=args.std_tr,
        pose_init=args.pose_init,
        dataset=""
    )

    num_sample = len(train_set)
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    
    if args.epoch_size == 0:
        args.epoch_size = len(train_set)
    # for i in range(num_sample):
    #     print(train_set[i])
    
    # create model
    print("=> creating model")
    pose_net = PoseNet(args.nlabel, args.std_tr, args.std_rot, add_geo_cost=args.geo, depth_augment=False)
    pose_net.init_weights()

    print('=> setting adam solver')

    optimizer = optimizers.Adam(args.lr, beta_1=args.momentum, beta_2=args.beta)
    
    with open(args.save_path / args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path / args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss'])

    for epoch in range(args.epochs):
        learning_rate = args.lr * (0.8 ** (epoch // 10))
        optimizer.learning_rate.assign(learning_rate)

        train_loss = train(args, train_set, pose_net, optimizer, training_writer, num_sample)

        if epoch % 10 == 0:
            pose_net.save_weights(args.save_path)
        
        with open(args.save_path / args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss])


def train(args, train_loader, pose_net, optimizer, train_writer, num_sample):
    """
    Train loader contains training examples
    Pose Net is the main model to be trained
    Optimizer passes the Adam optimizer
    Train writer is SummaryWriter object for TensorBoardX


    """

    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    
    end = time.time()

    for i, (tgt_img, ref_imgs, ref_poses, intrinsics, intrinsics_inv, tgt_depth, ref_depths, ref_noise_poses, initial_pose) in enumerate(train_loader):
        data_time.update(time.time() - end)
        tgt_img_var = tf.Variable(tgt_img)
        ref_imgs_var = [tf.Variable(img) for img in ref_imgs]
        ref_poses_var = [tf.Variable(pose) for pose in ref_poses]
        ref_noise_poses_var = [tf.Variable(pose) for pose in ref_noise_poses]
        initial_pose_var = tf.Variable(initial_pose)

        ref_depths_var = [tf.Variable(dep) for dep in ref_depths]
        intrinsics_var = tf.Variable(intrinsics)
        intrinsics_inv_var = tf.Variable(intrinsics_inv)
        tgt_depth_var = tf.Variable(tgt_depth)

        pose = tf.concat(ref_poses_var, 1)
        noise_pose = tf.concat(ref_noise_poses_var, 1)
        pose_norm = tf.norm(noise_pose[:, :, :3, 3], axis=-1, keepdims=True)  # b * n * 1
        

        with tf.GradientTape() as tape:
            # get PoseNet model output (pose angle, pose translation, rotation, translation)
            p_angle, p_trans, rot_c, trans_c = pose_net(tgt_img_var, ref_imgs_var, initial_pose_var, noise_pose,
                                                    intrinsics_var, intrinsics_inv_var,
                                                    tgt_depth_var,
                                                    ref_depths_var, trans_norm=pose_norm, training=True)

            batch_size = p_angle.shape[0]
            
            p_angle_v0 = tf.reshape(Softmax(axis=1)(p_angle), [batch_size, -1, 1])
            p_angle_v = tf.reduce_sum(p_angle_v0 * tf.cast(rot_c, tf.float32) , axis=1)
            
            p_trans_v0 = tf.reshape(Softmax(axis=1)(p_trans), [batch_size, -1, 1])
            p_trans_v = tf.reduce_sum(p_trans_v0 * tf.cast(trans_c, tf.float32) , axis=1)
            p_matrix = np.zeros((batch_size, 4, 4), dtype=np.float32)
            
            p_matrix[:, 3, 3] = 1
            p_matrix[:, :3, :] = np.concatenate([angle2matrix(p_angle_v).numpy(), tf.expand_dims(p_trans_v, -1).numpy()], axis=-1)
            p_matrix = tf.Variable(p_matrix)

            loss = 0.
            loss_rot = 0.
            loss_trans = 0.
            for j in range(len(ref_imgs)):
                exp_pose = tf.matmul(inv(pose[:, j]), noise_pose[:, j])
                gt_angle = matrix2angle(exp_pose[:, :3, :3])
                gt_trans = exp_pose[:, :3, 3]

                lambda_rot = 50     # weight parameter for rotation
                lambda_trans = 50     # weight parameter for translation
                
                loss_rot = lambda_rot * MeanAbsoluteError()(p_angle_v, gt_angle)
                loss_trans = lambda_trans * MeanAbsoluteError()(p_trans_v / pose_norm[:, :, 0], gt_trans / pose_norm[:, :, 0])

                loss = loss + loss_rot + loss_trans

        grads = tape.gradient(loss, pose_net.trainable_weights)
        optimizer.apply_gradients(zip(grads, pose_net.trainable_weights))
        
        if i > 0 and n_iter % args.print_freq == 0:
            train_writer.add_scalar('total_loss', loss.item(), n_iter)

        if n_iter > 0 and n_iter % 2000 == 0:
            pose_net.save_weights(args.save_path)
                # record loss and EPE
        
        losses.update(loss, batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        print(loss, loss_rot, loss_trans)
        
        with open(args.save_path / args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss])
        # import pdb;pdb.set_trace()
        if i % args.print_freq == 0:
            print(
                'Train {}: Time {} Data {} Loss: {:.4f} rot: {:.4f}trans: {:.4f}' \
                    .format(i, batch_time, data_time, loss, loss_rot,
                            loss_trans))
        n_iter += 1
    return losses.avg[0]


if __name__ == '__main__':
    main()
