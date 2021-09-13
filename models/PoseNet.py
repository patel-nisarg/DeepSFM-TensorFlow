import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, AveragePooling3D, Dense, ZeroPadding3D
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.layers.convolutional import Conv2DTranspose
from tensorflow_addons.layers import AdaptiveAveragePooling3D

from convert import *
from inverse_warp import inverse_warp_cost, depth_warp_cost
from models.submodule import *


class PoseNet(Model):
    def __init__(self, nlabel, std_tr, std_rot, add_geo_cost=False, depth_augment=False):
        super(PoseNet, self).__init__()
        self.nlabel = int(nlabel)
        self.std_tr = std_tr
        self.std_rot = std_rot
        self.add_geo = add_geo_cost
        self.depth_augmnet = depth_augment
        self.feature_extraction = feature_extraction(pool=True)

        if add_geo_cost:
            self.n_dres0 = Sequential([
                convbn_3d(32, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
                LeakyReLU(alpha=0.01), # a = 0.01 is default in torch but a=0.3 in tf
                conv_3d(64, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
                LeakyReLU(alpha=0.01),
                AveragePooling3D((4, 2, 2), strides=(4, 2, 2)),
                convbn_3d(128, 3, 1, 1),
                LeakyReLU(0.01)
            ])
            self.n_dres0_trans = Sequential([
                convbn_3d(64, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
                LeakyReLU(0.01),
                AveragePooling3D((4, 2, 2), (4, 2, 2)),
                convbn_3d(128, 3, 1, 1),
                LeakyReLU(0.01)
            ])

        else:

            self.dres0 = Sequential([
                conv_3d(32, 3, 1, 1),
                LeakyReLU(0.01),
                AveragePooling3D((1, 2, 2), strides=(1, 2, 2)),
                convbn_3d(32, 3, 1, 1),
                LeakyReLU(0.01)
            ])

            self.dres0_trans = Sequential([
                conv_3d(32, 3, 1, 1),
                LeakyReLU(0.01),
                AveragePooling3D((1, 2, 2), strides=(1, 2, 2)),
                convbn_3d(32, 3, 1, 1),
                LeakyReLU(0.01)
            ])

        self.dres1 = Sequential([
            conv_3d(128, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
            LeakyReLU(0.01),
            convbn_3d(128, 3, 1, 1)
        ])

        self.dres1_trans = Sequential([
            convbn_3d(128, (7,  3, 3), (1, 1, 1), (3, 1, 1)),
            LeakyReLU(0.01),
            convbn_3d(128, 3, 1, 1)
        ])

        self.dres2 = Sequential([
            conv_3d(128, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
		    LeakyReLU(0.01),
		    convbn_3d(128, 3, 1, 1)
        ])
        
        self.dres2_trans = Sequential([
            conv_3d(128, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
		    LeakyReLU(0.01),
		    convbn_3d(128, 3, 1, 1)
        ])

        self.AvgPool3d = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))
        self.dres3 = Sequential([
            conv_3d(128, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
            LeakyReLU(0.01),
            convbn_3d(128, 3, 1, 1)
        ])

        self.dres3_trans = Sequential([
            convbn_3d(128, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
            LeakyReLU(0.01),
            convbn_3d(128, 3, 1, 1)
        ])

        self.dres4 = Sequential([
            AveragePooling3D((2, 2 , 2), strides=(2, 2, 2)),
            convbn_3d(128, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
            LeakyReLU(0.01),
            convbn_3d(128, 3, 1, 1)
        ])
        
        self.dres4_trans = Sequential([
            convbn_3d(128, (7, 3, 3), (1, 1, 1), (3, 1, 1)),
            LeakyReLU(0.01),
            convbn_3d(128, 3, 1, 1)
        ])

        self.classify = Sequential([
            convbn_3d(512, (3, 1, 1), (1, 1, 1), (1, 0, 0)),
            LeakyReLU(0.01),
            AveragePooling3D((2, 2, 2), strides=(2, 2, 2)),
            convbn_3d(512, (1, 1, 1), (1, 1, 1), 0),
            LeakyReLU(0.01),
            ZeroPadding3D(((1, 1), (0, 0), (0, 0))),
            AdaptiveAveragePooling3D((16, 1, 1))
        ])
        self.fc = Dense(self.nlabel ** 3)
        self.classify_trans = Sequential([
            convbn_3d(512, (3, 1, 1), (1, 1, 1), (1, 0, 0)),
            LeakyReLU(0.01),
            AveragePooling3D((2, 2, 2), (2, 2, 2)),
            convbn_3d(512, (1, 1, 1), (1, 1, 1), 0),
            LeakyReLU(0.01),
            ZeroPadding3D(((2, 1), (0, 0), (0, 0))),
            AdaptiveAveragePooling3D((16, 1, 1))])

        self.fc_trans = Dense(self.nlabel ** 3)

        for layer in self.layers:
            if isinstance(layer, Sequential):
                for sublayer in layer.layers:
                    self._init_weights(layer=sublayer)
            else:
                self._init_weights(layer)
    
    def _init_weights(self, layer):
        if isinstance(layer, Conv2D):
            n = layer.kernel_size[0] * layer.kernel_size[1] * layer.filters
            layer.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=np.sqrt(2. / n))
        elif isinstance(layer, Conv3D):
            n = layer.kernel_size[0] * layer.kernel_size[1] * layer.kernel_size[2] * layer.filters
            layer.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=np.sqrt(2. / n))

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, Conv2D) or isinstance(layer, Conv2DTranspose):
                layer.kernel_initializer = tf.keras.initializers.GlorotNormal()
    
    def get_geo_loss_cost(self, refimg_fea, ref_depth, targetimg_fea, targets_depth, pose, intrinsics4, intrinsics_inv4):
        b, h, w, c = refimg_fea.shape
        # NOTE: Large memory!
        targetimg_fea_t = inverse_warp_cost(targetimg_fea, ref_depth, pose, intrinsics4,
                                        intrinsics_inv4)  # bNNN * h * w * c
        targetimg_fea_t = tf.reshape(targetimg_fea_t, [b, -1, h, w, c])

        # targetimg_fea_t = tf.transpose(targetimg_fea_t, perm=[0, 2, 3, 1])

        n = targetimg_fea_t.shape[1]
        refimg_fea = tf.reshape(refimg_fea, [b, 1, h, w, c])
        refimg_fea = tf.tile(refimg_fea, [1, n, 1, 1, 1])
        if self.add_geo:
            projected_depth, warped_depth = depth_warp_cost(targets_depth, ref_depth, pose, intrinsics4, intrinsics_inv4)
            return tf.concat([refimg_fea, targetimg_fea_t, projected_depth, warped_depth], axis=4)
        return tf.concat([refimg_fea, targetimg_fea_t], axis=4)


    def sample_poses(self, ref_pose, trans_norm):
        batch_size = ref_pose.shape[0]
        ref_trans = tf.cast(ref_pose[:, :3, 3], tf.float64)

        trans = tf.cast(tf.Variable(np.array(range(int(-self.nlabel / 2), int(self.nlabel / 2)))), tf.float64)
        trans = - (trans + 0.5) / trans[0] * self.std_tr
        trans1 = tf.tile(tf.reshape(trans, [1, self.nlabel, 1, 1, 1]), [batch_size, 1, self.nlabel, self.nlabel, 1])
        trans2 = tf.tile(tf.reshape(trans, [1, 1, self.nlabel, 1, 1]), [batch_size, self.nlabel, 1, self.nlabel, 1])
        trans3 = tf.tile(tf.reshape(trans, [1, 1, 1, self.nlabel, 1]), [batch_size, self.nlabel, self.nlabel, 1, 1])
        trans_vol = tf.concat([trans1, trans2, trans3], axis=4) * tf.cast(tf.reshape(trans_norm, [batch_size, 1, 1, 1, 1]), tf.float64)
        trans_volume = tf.tile(tf.reshape(ref_trans, [batch_size, 1, 1, 1, 3]), [1, self.nlabel, self.nlabel, self.nlabel, 1]) + trans_vol

        trans_disp = tf.identity(ref_pose)
        trans_disp = tf.tile(tf.reshape(trans_disp, [batch_size, 1, 1, 1, 4, 4]), [1, self.nlabel, self.nlabel, self.nlabel, 1, 1])
        trans_disp = trans_disp.numpy()
        trans_disp[:, :, :, :, :3, 3] = trans_volume
        trans_disp = tf.cast(tf.Variable(trans_disp), tf.float64)

        trans_vol = tf.reshape(trans_vol, [batch_size, -1, 3])
        
        rot = tf.cast(tf.Variable(np.array(range(int(-self.nlabel / 2), int(self.nlabel / 2)))), tf.float64)
        rot = - (rot + 0.5) / rot[0] * self.std_rot
        rot1 = tf.tile(tf.reshape(rot, [1, self.nlabel, 1, 1, 1]), [batch_size, 1, self.nlabel, self.nlabel, 1])
        rot2 = tf.tile(tf.reshape(rot, [1, 1, self.nlabel, 1, 1]), [batch_size, self.nlabel, 1, self.nlabel, 1])
        rot3 = tf.tile(tf.reshape(rot, [1, 1, 1, self.nlabel, 1]), [batch_size, self.nlabel, self.nlabel, 1, 1])
        angle_vol = tf.concat([rot1, rot2, rot3], axis=4) # [b, 10, 10, 10, 3, 3]
        angle_matrix = angle2matrix(angle_vol) # [b, 10, 10, 10, 3, 3]
        ref_pose2 = tf.tile(tf.reshape(ref_pose[:, :3, :3], [batch_size, 1, 1, 1, 3, 3]), [1, self.nlabel, self.nlabel, self.nlabel, 1, 1]) # [2, 10, 10, 10, 3, 3]
        rot_volume = tf.matmul(angle_matrix, ref_pose2)
        angle_vol = tf.reshape(angle_vol, [batch_size, -1, 3])
        rot_disp = tf.identity(ref_pose)
        rot_disp = tf.tile(tf.reshape(rot_disp, [batch_size, 1, 1, 1, 4, 4]), [1, self.nlabel, self.nlabel, self.nlabel, 1, 1])
        rot_disp = rot_disp.numpy()
        rot_disp[:, :, :, :, :3, :3] = rot_volume  # b * n * n * n * 4 * 4

        return tf.cast(tf.Variable(rot_disp), tf.float64), trans_disp, angle_vol, trans_vol

    def __call__(self, ref, targets, ref_pose, tgt_poses, intrinsics, intrinsics_inv, ref_depth=None, targets_depths=None,
                gt_poses=None, mode=0, trans_norm=None, training=True):
        
        intrinsics4 = tf.identity(intrinsics)
        intrinsics_inv4 = tf.identity(intrinsics_inv).numpy()
        intrinsics4 = intrinsics4.numpy()
        intrinsics4[:, :2, :] = intrinsics4[:, :2, :] / 8
        intrinsics4 = tf.Variable(intrinsics4)
        intrinsics_inv4[:, :2, :2] = intrinsics_inv4[:, :2, :2] * 8
        intrinsics_inv4 = tf.Variable(intrinsics_inv4)

        refimg_fea = self.feature_extraction(ref)
        batch_size, fea_h, fea_w, channels = refimg_fea.shape

        rot_pose_vol, trans_pose_vol, angle_vol, trans_vol = self.sample_poses(ref_pose, trans_norm[:, 0])
        ref_depth = tf.expand_dims(ref_depth, axis=3) # [batch, h, w, c]
        ref_depth = tf.image.resize(ref_depth, size=[fea_h, fea_w])
        ref_depth = tf.squeeze(ref_depth, axis=3)

        for j, target in enumerate(targets):
            targetimg_fea = self.feature_extraction(target)

            tgt_depth = tf.expand_dims(targets_depths[j], axis=3)
            tgt_depth = tf.image.resize(tgt_depth, size=[fea_h, fea_w])
            tgt_depth = tf.squeeze(tgt_depth, axis=3)

            tgt_pose = tf.cast(tf.expand_dims(tgt_poses[:, j], axis=1), tf.float64)
            tgt_pose = tf.tile(tgt_poses, [1, self.nlabel * self.nlabel * self.nlabel, 1, 1])
            tgt_pose = tf.reshape(tgt_pose, [-1, 4, 4])

            rel_trans_vol = tf.matmul(tgt_pose, inv(tf.reshape(trans_pose_vol, [-1, 4, 4])))[:, :3, :4]
            rel_trans_vol = tf.reshape(rel_trans_vol, [batch_size, self.nlabel, self.nlabel, self.nlabel, 3, 4])

            rel_rot_vol = tf.matmul(tgt_pose, inv(tf.reshape(rot_pose_vol, [-1, 4, 4])))[:, :3, :4]
            rel_rot_vol = tf.reshape(rel_rot_vol, [batch_size, self.nlabel, self.nlabel, self.nlabel, 3, 4])

            if self.add_geo:
                trans_cost = self.get_geo_loss_cost(refimg_fea, ref_depth, targetimg_fea, tgt_depth,
                                                    rel_trans_vol, intrinsics4, intrinsics_inv4)  # B*NNN
                rot_cost = self.get_geo_loss_cost(refimg_fea, ref_depth, targetimg_fea, tgt_depth,
                                                rel_rot_vol, intrinsics4, intrinsics_inv4)  # B*NNN
            else:
                rot_cost = self.get_geo_loss_cost(refimg_fea, ref_depth, targetimg_fea, None,
                                                rel_rot_vol, intrinsics4, intrinsics_inv4)  # B*NNN
                trans_cost = self.get_geo_loss_cost(refimg_fea, ref_depth, targetimg_fea, None,
                                                    rel_trans_vol, intrinsics4, intrinsics_inv4)  # B*NNN
            
            if mode % 2 == 0:
                if self.add_geo:
                    trans_cost0 = self.n_dres0_trans(trans_cost)
                else:
                    trans_cost0 = self.dres0_trans(trans_cost)

                trans_cost0 = self.dres1_trans(trans_cost0) + trans_cost0
                trans_cost0 = self.dres2_trans(trans_cost0) + trans_cost0
                trans_cost0 = self.dres3_trans(trans_cost0) + trans_cost0
                trans_cost0 = self.dres4_trans(trans_cost0) + trans_cost0
                trans_cost0 = self.classify_trans(trans_cost0)

            if mode < 2:
                if self.add_geo:
                    rot_cost0 = self.n_dres0(rot_cost)
                else:
                    rot_cost0 = self.dres0(rot_cost)
                rot_cost0 = self.dres1(rot_cost0) + rot_cost0
                rot_cost0 = self.dres2(rot_cost0) + rot_cost0
                rot_cost0 = self.dres3(rot_cost0) + rot_cost0
                rot_cost0 = self.dres4(rot_cost0) + self.AvgPool3d(rot_cost0)
                rot_cost0 = self.classify(rot_cost0)
            if j == 0:
                trans_costs = trans_cost0
                rot_costs = rot_cost0
            else:
                trans_costs = trans_cost0 + trans_costs
                rot_costs = rot_costs + rot_cost0

        if mode % 2 == 0:
            trans_costs = (trans_costs / len(targets))
            trans_costs = tf.reshape(trans_costs, [batch_size, -1])
            trans_costs = self.fc_trans(trans_costs)

            pred_trans = trans_costs
        
        if mode < 2:
            rot_costs = (rot_costs / len(targets))
            rot_costs = tf.reshape(rot_costs, [batch_size, -1])

            rot_costs = self.fc(rot_costs)
            pred_rot = rot_costs
        
        return pred_rot, pred_trans, angle_vol, trans_vol
