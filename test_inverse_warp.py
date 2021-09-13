from enum import Flag

from tensorflow.python.ops.array_ops import zeros
import inverse_warp
import tensorflow as tf
import numpy as np


b, _, h, w = (5, 4, 3, 3)
cam_coords = tf.Variable([[[[ 0.4388,  0.5580,  0.8605],
        [ 0.7867,  0.8700,  0.1743],
        [ 0.9503,  0.2236,  0.3671]],

        [[ 0.5550,  0.4304,  0.6571],
        [ 0.1747,  0.3535,  0.7084],
        [ 0.1074,  0.9887,  0.3613]],

        [[ 0.3223,  0.7353,  0.6976],
        [ 0.5262,  0.2941,  0.5787],
        [ 0.7681,  0.3910,  0.8915]]],


    [[[ 0.9174,  0.2740,  0.0466],
        [ 0.4776,  0.1006,  0.2885],
        [ 0.4829,  0.4482,  0.9964]],

        [[ 0.4861,  0.5131,  0.7166],
        [ 0.4080,  0.3395,  0.6807],
        [ 0.0843,  0.3618,  0.2157]],

        [[ 0.6358,  0.3774,  0.5343],
        [ 0.9737,  0.9437,  0.6649],
        [ 0.8482,  0.6590,  0.0231]]],


    [[[ 0.3208,  0.6365,  0.2254],
        [ 0.0108,  0.1798,  0.8895],
        [ 0.9377,  0.7684,  0.6588]],

        [[ 0.4233,  0.1134,  0.7098],
        [ 0.9613,  0.1413,  0.9021],
        [ 0.0463,  0.3654,  0.4208]],

        [[ 0.0503,  0.7790,  0.8203],
        [ 0.7009,  0.7641,  0.4171],
        [ 0.9194,  0.6246,  0.7451]]],


    [[[ 0.7691,  0.9128,  0.1138],
        [ 0.6835,  0.0144,  0.6097],
        [ 0.2681,  0.1875,  0.9824]],

        [[ 0.3106,  0.5122,  0.4597],
        [ 0.7126,  0.3018,  0.6232],
        [ 0.6993,  0.6925,  0.5170]],

        [[ 0.5602,  0.1837,  0.1634],
        [ 0.8448,  0.9167,  0.0335],
        [ 0.3141,  0.4877,  0.1924]]],


    [[[ 0.9786,  0.6795,  0.3119],
        [ 0.9387,  0.1088,  0.8992],
        [ 0.1995,  0.3654,  0.4065]],

        [[ 0.5189,  0.0208,  0.2392],
        [ 0.2072,  0.4923,  0.0955],
        [ 0.3865,  0.5324,  0.8767]],

        [[ 0.7267,  0.3141,  0.8549],
        [ 0.2955,  0.1860,  0.8867],
        [ 0.5864,  0.2437,  0.9652]]]])

proj_c2p_rot = tf.Variable([[[ 0.9519,  0.7807,  0.4622],
        [ 0.8027,  0.9605,  0.3106],
        [ 0.6795,  0.7355,  0.0708]],

    [[ 0.7386,  0.8554,  0.0837],
        [ 0.0624,  0.9268,  0.5184],
        [ 0.7072,  0.3359,  0.1243]],

    [[ 0.8852,  0.2759,  0.2909],
        [ 0.9471,  0.2859,  0.1512],
        [ 0.5040,  0.4028,  0.9223]],

    [[ 0.7973,  0.7891,  0.4355],
        [ 0.2714,  0.6797,  0.3326],
        [ 0.6152,  0.8880,  0.0522]],

    [[ 0.4240,  0.3652,  0.8090],
        [ 0.1669,  0.2424,  0.6919],
        [ 0.0496,  0.1282,  0.9153]]])

proj_c2p_tr = tf.Variable([[[ 0.7491],
        [ 0.2774],
        [ 0.7433]],

    [[ 0.0682],
        [ 0.5410],
        [ 0.7853]],

    [[ 0.0075],
        [ 0.1930],
        [ 0.4513]],

    [[ 0.6087],
        [ 0.1779],
        [ 0.3777]],

    [[ 0.9447],
        [ 0.7430],
        [ 0.3217]]])

# cam2pixel function
# cam_coords_flat = tf.reshape(cam_coords, [b, 3, -1])  # [B, 3, H*W]
# if proj_c2p_rot is not None:
#     pcoords = tf.matmul(proj_c2p_rot, cam_coords_flat)
# else:
#     pcoords = cam_coords_flat

# if proj_c2p_tr is not None:
#     pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
# X = pcoords[:, 0]
# Y = pcoords[:, 1]
# Z = pcoords[:, 2]
# Z = tf.clip_by_value(Z, clip_value_min=1e-3, clip_value_max=tf.float32.max)

# rounded = False
# if rounded:
#     X_norm = tf.math.round(2 * (X / Z)) / (
#                 w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
#     Y_norm = tf.math.round(2 * (Y / Z)) / (h - 1) - 1  # Idem [B, H*W]
# else:
#     X_norm = 2 * (X / Z) / (
#                 w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
#     Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]

# padding_mode = 'zeros'
# if padding_mode == 'zeros':
#     X_mask = (X_norm.numpy() > 1) + (X_norm.numpy() < -1)
#     X_norm = X_norm.numpy()
#     X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
#     X_norm = tf.Variable(X_norm)
#     Y_mask = (Y_norm.numpy() > 1) + (Y_norm.numpy() < -1)
#     Y_norm = Y_norm.numpy()
#     Y_norm[Y_mask] = 2
#     Y_norm = tf.Variable(Y_norm)

# pixel_coords = tf.stack([X_norm, Y_norm], axis=2)
# pixel_coords = tf.reshape(pixel_coords, [b, h, w, 2])

# print(pixel_coords)


#cam2pixel_cost function
# cam_coords_flat = tf.reshape(cam_coords, [b, 3, -1])
# pcoords = tf.matmul(proj_c2p_rot, tf.reshape(cam_coords_flat, [b, 1, 3, h * w]))

# if proj_c2p_tr is not None:
#     pcoords = pcoords + proj_c2p_tr
# X = pcoords[:, :, 0]
# Y = pcoords[:, :, 1]
# Z = pcoords[:, :, 2]
# Z = tf.clip_by_value(Z, clip_value_min=1e-3, clip_value_max=tf.float32.max)

# X_norm = 2 * (X / Z) / (
#                 w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
# Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]

# padding_mode = 'zeros'
# if padding_mode == 'zeros':
#     X_mask = (X_norm.numpy() > 1) + (X_norm.numpy() < -1)
#     X_norm = X_norm.numpy()
#     X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
#     X_norm = tf.Variable(X_norm)

#     Y_mask = (Y_norm.numpy() > 1) + (Y_norm.numpy() < -1)
#     Y_norm = Y_norm.numpy()
#     Y_norm[Y_mask] = 2
#     Y_norm = tf.Variable(Y_norm)


# pixel_coords = tf.stack([X_norm, Y_norm], axis=3)

# pixel_coords = tf.reshape(pixel_coords, [b, -1, h, w, 2])
# print(pixel_coords, b, h, w)


# cam2depth function
# cam_coords_flat = tf.reshape(cam_coords, [b, 3, -1])
# if proj_c2p_rot is not None:
#     pcoords = tf.matmul(proj_c2p_rot, cam_coords_flat)
# else:
#     pcoords = cam_coords_flat

# if proj_c2p_tr is not None:
#     pcoords = pcoords + proj_c2p_tr

# z = pcoords[:, 2, :]
# z = tf.reshape(z, [b, h, w])
# print(z)

cam_coords = tf.Variable([[[[ 0.1287,  0.3675,  0.4789],
          [ 0.6742,  0.5734,  0.8497],
          [ 0.4899,  0.9304,  0.5811]],

         [[ 0.6890,  0.6171,  0.1087],
          [ 0.8574,  0.9804,  0.9492],
          [ 0.2728,  0.5114,  0.8681]],

         [[ 0.5495,  0.0710,  0.0520],
          [ 0.0299,  0.4845,  0.8421],
          [ 0.5947,  0.0430,  0.4589]]],


        [[[ 0.2429,  0.7937,  0.5121],
          [ 0.4035,  0.7505,  0.2552],
          [ 0.4364,  0.4730,  0.1659]],

         [[ 0.1078,  0.3605,  0.6697],
          [ 0.4914,  0.1776,  0.2659],
          [ 0.6438,  0.0530,  0.2829]],

         [[ 0.9296,  0.0562,  0.9552],
          [ 0.6903,  0.4702,  0.1100],
          [ 0.8747,  0.3482,  0.3108]]],


        [[[ 0.3012,  0.3978,  0.1216],
          [ 0.3576,  0.8265,  0.4124],
          [ 0.2895,  0.1552,  0.5792]],

         [[ 0.4626,  0.6545,  0.4895],
          [ 0.4832,  0.6171,  0.0997],
          [ 0.9776,  0.7485,  0.4112]],

         [[ 0.3792,  0.2275,  0.3570],
          [ 0.4564,  0.3900,  0.1874],
          [ 0.9577,  0.3262,  0.4588]]],


        [[[ 0.4014,  0.4391,  0.7987],
          [ 0.2578,  0.4109,  0.5482],
          [ 0.3636,  0.6297,  0.3052]],

         [[ 0.5134,  0.0474,  0.1486],
          [ 0.2988,  0.1967,  0.2107],
          [ 0.0638,  0.3427,  0.0204]],

         [[ 0.5423,  0.0052,  0.8679],
          [ 0.3388,  0.2809,  0.4133],
          [ 0.8407,  0.8706,  0.0577]]],


        [[[ 0.6656,  0.9604,  0.9343],
          [ 0.7144,  0.7200,  0.9984],
          [ 0.3114,  0.3599,  0.6416]],

         [[ 0.4428,  0.1489,  0.0106],
          [ 0.4488,  0.5937,  0.0571],
          [ 0.3358,  0.1230,  0.0320]],

         [[ 0.1069,  0.9444,  0.9588],
          [ 0.8189,  0.9020,  0.5041],
          [ 0.5626,  0.7810,  0.6482]]],


        [[[ 0.1794,  0.5291,  0.1796],
          [ 0.4960,  0.5314,  0.3825],
          [ 0.7500,  0.1112,  0.8317]],

         [[ 0.3741,  0.9305,  0.4609],
          [ 0.7229,  0.6842,  0.6279],
          [ 0.3033,  0.6519,  0.2081]],

         [[ 0.0951,  0.0955,  0.8864],
          [ 0.4114,  0.9103,  0.0874],
          [ 0.4518,  0.9038,  0.2441]]],


        [[[ 0.2726,  0.6881,  0.0007],
          [ 0.4238,  0.6602,  0.2700],
          [ 0.1672,  0.6152,  0.9839]],

         [[ 0.5514,  0.3817,  0.9980],
          [ 0.5061,  0.9367,  0.4298],
          [ 0.1778,  0.4002,  0.0924]],

         [[ 0.0065,  0.5317,  0.1609],
          [ 0.4287,  0.4411,  0.7354],
          [ 0.3882,  0.8437,  0.7831]]],


        [[[ 0.1032,  0.6014,  0.1067],
          [ 0.5252,  0.9641,  0.7270],
          [ 0.4743,  0.2761,  0.3654]],

         [[ 0.3674,  0.1988,  0.7238],
          [ 0.2656,  0.9630,  0.4317],
          [ 0.2040,  0.7162,  0.0666]],

         [[ 0.7888,  0.0717,  0.2711],
          [ 0.2954,  0.7700,  0.0032],
          [ 0.8042,  0.8782,  0.4965]]],


        [[[ 0.4479,  0.2166,  0.6911],
          [ 0.3651,  0.4819,  0.1391],
          [ 0.6167,  0.3820,  0.5450]],

         [[ 0.7430,  0.1379,  0.9110],
          [ 0.4481,  0.1382,  0.9716],
          [ 0.2027,  0.5966,  0.1328]],

         [[ 0.9989,  0.5558,  0.1212],
          [ 0.9786,  0.4844,  0.8003],
          [ 0.5953,  0.0550,  0.9349]]],


        [[[ 0.0614,  0.0183,  0.6570],
          [ 0.1342,  0.3923,  0.8931],
          [ 0.1227,  0.9234,  0.6132]],

         [[ 0.1016,  0.6805,  0.6162],
          [ 0.0846,  0.5049,  0.5889],
          [ 0.6272,  0.9747,  0.3504]],

         [[ 0.9706,  0.6121,  0.1996],
          [ 0.1683,  0.4575,  0.1458],
          [ 0.8589,  0.9166,  0.9518]]]])


b, _, h, w = cam_coords.shape
n = proj_c2p_rot.shape[1]
cam_coords_flat = tf.reshape(cam_coords, [b, 3, -1])  # [B, 3, H*W]
# if proj_c2p_rot is not None:
pcoords = tf.matmul(proj_c2p_rot, tf.reshape(cam_coords_flat, [b, 1, 3, h * w]))  # b, nnn, 3, h*w
# else:
#     pcoords = cam_coords_flat

if proj_c2p_tr is not None:
    pcoords = pcoords + proj_c2p_tr  # b, nnn, 3, h*w
z = pcoords[:, :, 2, :]
print(z.shape)
z = tf.reshape(z, [b, n, h, w])
print(z)
