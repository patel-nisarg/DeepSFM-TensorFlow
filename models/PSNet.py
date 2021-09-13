import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, ZeroPadding3D, ZeroPadding2D, Conv2DTranspose, UpSampling3D, Softmax
from tensorflow.keras.models import Sequential, Model

from inverse_warp import inverse_warp, depth_warp
from models.submodule import *


def convtext(filters, kernel_size=3, stride=1, dilation=1):
    padding = (kernel_size - 1) * dilation // 2
    return Sequential([
        ZeroPadding2D(padding),
        Conv2D(filters, kernel_size=kernel_size, strides=stride, dilation_rate=dilation),
        LeakyReLU(alpha=0.1)
    ])

class PSNet(Model):
    def __init__(self, nlabel, mindepth, add_geo_cost=False, depth_augment=False):
        super(PSNet, self).__init__()
        self.nlabel = nlabel
        self.mindepth = mindepth
        self.add_geo = add_geo_cost
        self.depth_augmnet = depth_augment
        self.feature_extraction = feature_extraction()

        self.convs = Sequential([
            convtext(128, 3, 1, 1),
            convtext(128, 3, 1, 2),
            convtext(128, 3, 1, 4),
            convtext(96, 3, 1, 8),
            convtext(64, 3, 1, 16),
            convtext(32, 3, 1, 1),
            convtext(1, 3, 1, 1)
        ])
        if add_geo_cost:
            self.n_dres0 = Sequential([
                convbn_3d_o(32, 3, 1, 1),
                ReLU(),
                convbn_3d_o(32, 3, 1, 1),
                ReLU()
            ])
        else:
            self.dres0 = Sequential([
                convbn_3d_o(32, 3, 1, 1),
                ReLU(),
                convbn_3d_o(32, 3, 1, 1),
                ReLU
            ])

        self.dres1 = Sequential([
            convbn_3d_o(32, 3, 1, 1),
            ReLU(),
            convbn_3d_o(32, 3, 1, 1)
        ])

        self.dres2 = Sequential([
            convbn_3d_o(32, 3, 1, 1),
            ReLU(),
            convbn_3d_o(32, 3, 1, 1)
        ])

        self.dres3 = Sequential([
            convbn_3d_o(32, 3, 1, 1),
            ReLU(),
            convbn_3d_o(32, 3, 1, 1)
        ])

        self.dres4 = Sequential([
            convbn_3d_o(32, 3, 1, 1),
            ReLU(),
            convbn_3d_o(32, 3, 1, 1)
        ])

        self.classify = Sequential([
            convbn_3d_o(32, 32, 3, 1),
            ReLU(),
            ZeroPadding3D(1),
            Conv3D(1, 3, 1, 'valid')
        ])

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

    def __call__(self, ref, targets, pose, intrinsics, intrinsics_inv, training, targets_depth=None, mindepth=0.5):

        intrinsics4 = tf.identity(intrinsics)
        intrinsics_inv4 = tf.identity(intrinsics_inv).numpy()
        intrinsics4 = intrinsics4.numpy()
        intrinsics4[:, :2, :] = intrinsics4[:, :2, :] / 4
        intrinsics4 = tf.Variable(intrinsics4)
        intrinsics_inv4[:, :2, :2] = intrinsics_inv4[:, :2, :2] * 4
        intrinsics_inv4 = tf.Variable(intrinsics_inv4)

        refimg_fea = self.feature_extraction(ref)

        disp2depth = tf.Variable(tf.ones([refimg_fea.shape[0], refimg_fea[2], refimg_fea[3]])) * self.mindepth * self.nlabel

        for j, target in enumerate(targets):
            if self.add_geo:
                cost = tf.Variable(tf.zeros([refimg_fea.shape[0], refimg_fea.shape[1] * 2 + 2, self.nlabel,
                                            refimg_fea.shape[2], refimg_fea[3]]))
            else:
                cost = tf.Variable(tf.zeros([refimg_fea.shape[0], refimg_fea[1] * 2, self.nlabel,
                                            refimg_fea.shape[2], refimg_fea[3]]))
            targetimg_fea = self.feature_extraction(target)
            if self.depth_augment:
                noise = tf.cast(tf.Variable(np.random.normal(loc=0.0, scale=mindepth / 10, size=(1, 240, 320))), tf.float64)
            else:
                noise = 0
            for i in range(self.nlabel):
                depth = tf.divide(disp2depth, i + 1e-16)
                targetimg_fea_t = inverse_warp(targetimg_fea, depth, pose[:, j], intrinsics4, intrinsics_inv4)
                if self.add_geo:
                    assert targets_depth is not None
                    projected_depth, warped_depth = depth_warp(targets_depth[j] + noise, depth, pose[:, j], intrinsics4, intrinsics_inv4)
                    cost = cost.numpy()
                    cost[:, -2, i, :, :] = projected_depth
                    cost[:, -1, i, :, :] = warped_depth
                    cost = tf.Variable(cost)
                cost = cost.numpy()
                cost[:, :refimg_fea.shape[1], i, :, :] = refimg_fea
                cost[:, refimg_fea[1]:refimg_fea.shape[1] * 2, i, :, :] = targetimg_fea_t
                cost = tf.Variable(cost)

            if self.add_geo:
                cost = self.ndres0(cost)
            else:
                cost0 = self.dres0(cost)
            cost0 = self.dres1(cost0) + cost0
            cost0 = self.dres2(cost0) + cost0
            cost0 = self.dres3(cost0) + cost0
            cost0 = self.dres4(cost0) + cost0
            cost0 = self.classify(cost0)
            if j == 0:
                costs = cost0
            else:
                costs = costs + cost0

        costs = costs / len(targets)

        costss = tf.Variable(tf.zeros(refimg_fea.shape[0], 1, self.nlabel, refimg_fea.shape[2],
                                    refimg_fea.shape[3]))
        for i in range(self.nlabel):
            costt = costs[:, :, i, :, :]
            costss = costss.numpy()
            costss[:, :, i, :, :] = self.convs(tf.concat([refimg_fea, costt], axis=1)) + costt
            costss = tf.Variable(costss)
        
        costs = UpSampling3D([self.nlabel, ref.shape[2], ref.shape[3]])(costs)
        costs = tf.squeeze(costs, axis=1)
        pred0 = Softmax(axis=1)(costs)
        pred0 = disparityregression(self.nlabel)(pred0)
        depth0 = self.mindepth * self.nlabel / (tf.expand_dims(pred0, axis=1 + 1e-16))

        costss = UpSampling3D([self.nlabel, ref.shape[2], ref.shape[3]])(costss)
        costss = tf.expand_dims(costss, axis=1)

        pred = Softmax(axis=1)(costss)
        pred = disparityregression(self.nlabel)(pred)
        depth = self.mindepth * self.nlabel / (tf.expand_dims(pred, axis=1 + 1e-16))

        if training:
            return depth0, depth
        else:
            return depth
