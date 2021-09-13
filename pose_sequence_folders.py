import os
import re
from tensorflow.keras.utils import Sequence
import numpy as np
from imageio import imread
from path import Path
import random
from tensorflow.python.ops.array_ops import stack
from transforms3d.euler import *
import skimage.io as io
from tqdm import tqdm
import tensorflow as tf

# https://keras.io/examples/vision/oxford_pets_image_segmentation/
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

def load_as_float(path):
    return imread(path).astype(np.float32)


def read_png_depth(path):
    depth = io.imread(path).astype(np.float32)
    depth = depth / 1000
    return depth


def batch_samples(sample_list, batch_size):
    """
    Batches list of samples into chunks of size 'batch_size'.
    """
    for i in range(0, len(sample_list), batch_size):
        yield sample_list[i:i + batch_size]


class SequenceFolder(Sequence):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, batch_size=6, seed=None, ttype='train.txt', sequence_length=2, transform=None,
                 add_geo=False, depth_source="p", dataset="", gt_source='g', std=0.2,
                 pose_init=None, get_path=False):
        print(dataset)
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root / ttype
        scenes = [self.root / folder[:-1] for folder in open(scene_list_path) if folder.startswith(dataset)]
        self.ttype = ttype
        self.std = std
        self.scenes = sorted(scenes)
        self.transform = transform
        self.geo = add_geo
        self.gt_source = gt_source
        self.depth_source = depth_source
        self.get_path = get_path
        self.batch_size = batch_size
        if not pose_init:
            pass
        else:
            self.pose_init = pose_init
            self.crawl_folders_pose(sequence_length)

    def crawl_folders_pose(self, sequence_length):
        self.sequence_set = []
        demi_length = sequence_length // 2
        p_num = 0
        g_num = 0
        for scene in tqdm(self.scenes):
            intrinsics = np.genfromtxt(scene / 'cam.txt').astype(np.float32).reshape((3, 3))
            poses = np.genfromtxt(scene / 'poses.txt').astype(np.float32)
            poses_pd = np.genfromtxt(scene + '/%s_poses.txt' % self.pose_init).astype(np.float32)
            imgs = sorted(scene.files('*.jpg'))

            if len(imgs) < sequence_length:
                continue
            for i in range(len(imgs)):
                if i < demi_length:
                    shifts = list(range(0, sequence_length))
                    shifts.pop(i)
                elif i >= len(imgs) - demi_length:
                    shifts = list(range(len(imgs) - sequence_length, len(imgs)))
                    shifts.pop(i - len(imgs))
                else:
                    shifts = list(range(i - demi_length, i + (sequence_length + 1) // 2))
                    shifts.pop(demi_length)

                img = imgs[i]
                depth = img.dirname() / img.name[:-4] + '.npy'
                if self.gt_source == 'p':
                    depth = img.dirname() / img.name[:-4] + '_p.npy'
                elif self.gt_source != 'g':
                    depth = img.dirname() / img.name[:-4] + '_' + self.depth_source + '.npy'

                pose_tgt = np.concatenate((poses[i, :].reshape((3, 4)), np.array([[0, 0, 0, 1]])), axis=0)
                pose_tgt_pd = np.concatenate((poses_pd[i, :].reshape((3, 4)), np.array([[0, 0, 0, 1]])), axis=0)
                initial_pose = np.eye(4).astype(np.float32)
                sample = {'intrinsics': intrinsics, 'tgt': img, 'tgt_depth': depth, 'ref_imgs': [],
                          'ref_poses': [], 'ref_noise_poses': [], 'ref_noise_angles': [], 'initial_pose': initial_pose,
                          'ref_depths': [], 'ref_angles': [], 'scale_gt': 0}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[j])
                    if self.geo:
                        if self.depth_source == 'g':
                            sample['ref_depths'].append(imgs[j].dirname() / imgs[j].name[:-4] + '.npy')
                        elif self.depth_source == 'p':
                            path = imgs[j].dirname() / imgs[j].name[:-4] + '_p.npy'
                            # path = str(imgs[j].dirname()).replace(str(self.root), str(self.pose_init)) +'/'+ imgs[j].name[:-4] + '.depth.png'
                            if (os.path.exists(path)):
                                sample['ref_depths'].append(path)
                                p_num += 1
                            else:
                                sample['ref_depths'].append(imgs[j].dirname() / imgs[j].name[:-4] + '.npy')
                                g_num += 1
                        else:
                            path = imgs[j].dirname() / imgs[j].name[:-4] + '_' + self.depth_source + '.npy'
                            if (os.path.exists(path)):
                                sample['ref_depths'].append(path)
                                p_num += 1
                            else:
                                sample['ref_depths'].append(imgs[j].dirname() / imgs[j].name[:-4] + '.npy')
                                g_num += 1
                    pose_src = np.concatenate((poses[j, :].reshape((3, 4)), np.array([[0, 0, 0, 1]])), axis=0)
                    pose_rel = pose_src @ np.linalg.inv(pose_tgt)
                    pose_src_pd = np.concatenate((poses_pd[j, :].reshape((3, 4)), np.array([[0, 0, 0, 1]])), axis=0)
                    pose_rel_pd = pose_src_pd @ np.linalg.inv(pose_tgt_pd)
                    # scale
                    if sample['scale_gt'] <= 0:
                        # no scale
                        scale_gt = np.ones_like(np.linalg.norm(pose_rel[:3, 3]))
                    else:
                        scale_gt = sample['scale_gt']
                    pose_rel[:3, 3] = pose_rel[:3, 3] / scale_gt
                    pose_rel_pd[:3, 3] = pose_rel_pd[:3, 3] / scale_gt
                    sample['scale_gt'] = scale_gt

                    pose = pose_rel.reshape((1, 4, 4)).astype(np.float32)
                    pose_pd = pose_rel_pd.reshape((1, 4, 4)).astype(np.float32)
                    sample['ref_poses'].append(pose)

                    sample['ref_noise_poses'].append(pose_pd)

                self.sequence_set.append(sample)
        # if self.ttype == 'train.txt':
        #     random.shuffle(sequence_set)
        print("pn:", p_num, "  gn:", g_num)
        self.samples = list(batch_samples(self.sequence_set, self.batch_size))  # Split samples into even sized batches


    def __getitem__(self, index):
        # iterate through batch in samples to transform and save to list of tuples
        sample_batch = self.samples[index]
        batch_return = []
        for i in range(len(sample_batch)):
            sample = sample_batch[i]
            tgt_img = load_as_float(sample['tgt'])
            # tgt_depth = read_png_depth(sample['tgt_depth'])
            tgt_depth = np.load(sample['tgt_depth'])
            scale = sample['scale_gt']
            tgt_depth = tgt_depth / scale

            nanmask = tgt_depth != tgt_depth
            num = np.sum(nanmask)
            if num != 0:
                print('tgt depth nan')
            tgt_depth[nanmask] = 1
            ref_depths = []

            for path in sample['ref_depths']:
                # ref_depth = read_png_depth(path)
                ref_depth = np.load(path)
                ref_depth = ref_depth / scale
                nanmask = ref_depth != ref_depth
                num = np.sum(nanmask)
                if (num != 0):
                    print('ref depth nan')
                ref_depth[nanmask] = 1
                ref_depths.append(ref_depth)

            ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
            ref_poses = sample['ref_poses']
            ref_noise_poses = sample['ref_noise_poses']
            initial_pose = sample['initial_pose']
            if self.transform is not None:
                imgs, depths, intrinsics = self.transform([tgt_img] + ref_imgs, [tgt_depth] + ref_depths,
                                                        np.copy(sample['intrinsics']))
                tgt_img = imgs[0]
                tgt_depth = depths[0]
                ref_imgs = imgs[1:]
                ref_depths = depths[1:]

            else:
                intrinsics = np.copy(sample['intrinsics'])
            if self.get_path:
                batch_return.append([tgt_img, ref_imgs, ref_poses, intrinsics, np.linalg.inv(intrinsics), tgt_depth, ref_depths, \
                    ref_noise_poses, initial_pose, sample['tgt'], sample['ref_imgs']])
            else:
                batch_return.append([tgt_img, ref_imgs, ref_poses, intrinsics, np.linalg.inv(intrinsics), tgt_depth, ref_depths, \
                ref_noise_poses, initial_pose])
        
        tgt_img = batch_return[0][0]
        ref_imgs = batch_return[0][1]
        ref_poses = batch_return[0][2]
        intrinsics = batch_return[0][3]
        intrinsics_inv = batch_return[0][4]
        tgt_depth = batch_return[0][5]
        ref_depths = batch_return[0][6]
        ref_noise_poses = batch_return[0][7]
        initial_pose = batch_return[0][8] 
        
        if self.batch_size == 1:
            return tuple([tf.expand_dims(target, axis=0) for target in (tgt_img, ref_imgs, ref_poses, intrinsics, intrinsics_inv, tgt_depth, ref_depths, ref_noise_poses, initial_pose)])
        
        for _, sample in enumerate(batch_return[1:]):
            tgt_img = tf.stack([tgt_img, sample[0]], axis=0)
            ref_imgs =  [tf.stack(list(ref_img)) for ref_img in zip(ref_imgs, sample[1])]
            ref_poses = [tf.stack(list(ref_pose)) for ref_pose in zip(ref_poses, sample[2])]
            intrinsics = tf.stack([intrinsics, sample[3]], axis=0)
            intrinsics_inv = tf.stack([intrinsics_inv, sample[4]], axis=0)
            tgt_depth = tf.stack([tgt_depth, sample[5]], axis=0)
            ref_depths = [tf.stack(list(ref_depth)) for ref_depth in zip(ref_depths, sample[6])]
            ref_noise_poses = [tf.stack(list(noise_poses)) for noise_poses in zip(ref_noise_poses, sample[7])] # ref noise poses is a list of [Tensor(1, 4, 4), Tensor(1, 4, 4)], we want [(batch, 4, 4), (batch, 4, 4)]
            initial_pose = tf.stack([initial_pose, sample[8]], axis=0)
        
        return (tgt_img, ref_imgs, ref_poses, intrinsics, intrinsics_inv, tgt_depth, ref_depths, ref_noise_poses, initial_pose)


    def __len__(self):
        return len(self.samples)
