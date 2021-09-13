from __future__ import division
from re import I
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import Shape, shape


pixel_coords = None

def grid_sample(input, grid, mode, padding_mode):
	"""
	input: 4D tensor of shape (B, C, H, W)
	grid: 4D tensor of shape (B, H, W, 2)
	mode: interpolation mode to calculate output. "nearest" for nearest neighbour or "bilinear"
	padding_mode: "zeros":

	output: 4D tensor of shape (B, H, W, 2) 
	"""
	return input

def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.shape # 2, 30, 40
    
    i_range = tf.Variable(tf.broadcast_to(tf.reshape(tf.range(0, h), (1, h, 1)), [1, h, w]))
    i_range = tf.cast(i_range, depth.dtype)
    
    j_range = tf.Variable(tf.broadcast_to(tf.reshape(tf.range(0, w), (1, 1, w)), [1, h, w]))
    j_range = tf.cast(j_range, depth.dtype)
    
    ones = tf.Variable(tf.ones((1, h, w)))
    ones = tf.cast(ones, depth.dtype)
    pixel_coords = tf.stack((j_range, i_range, ones), axis=1)


def check_sizes(input, input_name, expected):
    condition = [tf.rank(input).numpy() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.shape[i] == int(size))
    assert (all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected),
	                                                                          list(input.shape))

def pixel2cam(depth, intrinsics_inv):
	global pixel_coords
	"""Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
	b, h, w = depth.shape
	if (pixel_coords is None) or pixel_coords.shape[2] < h:
		set_id_grid(depth)
	current_pixel_coords = tf.reshape(tf.broadcast_to(pixel_coords[:, :, :h, :w], [b, 3, h, w]), [b, 3, -1])
	current_pixel_coords = tf.cast(current_pixel_coords, depth.dtype)
	cam_coords = tf.reshape(tf.matmul(intrinsics_inv, current_pixel_coords), [b, 3, h, w])
	return cam_coords * tf.expand_dims(depth, axis=1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode, rounded=False):
	"""Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 3]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
	b, _, h, w = cam_coords.shape
	cam_coords_flat = tf.reshape(cam_coords, [b, -1, 3])  # [B, 3, H*W]
	if proj_c2p_rot is not None:
		pcoords = tf.matmul(proj_c2p_rot, cam_coords_flat)
	else:
		pcoords = cam_coords_flat

	if proj_c2p_tr is not None:
		pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
	X = pcoords[:, 0]
	Y = pcoords[:, 1]
	Z = pcoords[:, 2]
	Z = tf.clip_by_value(Z, clip_value_min=1e-3, clip_value_max=tf.float32.max)
	if rounded:
		X_norm = tf.math.round(2 * (X / Z)) / (
					w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
		Y_norm = tf.math.round(2 * (Y / Z)) / (h - 1) - 1  # Idem [B, H*W]
	else:
		X_norm = 2 * (X / Z) / (
					w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
		Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]

	if padding_mode == 'zeros':
		X_mask = (X_norm.numpy() > 1) + (X_norm.numpy() < -1)
		X_norm = X_norm.numpy()
		X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
		X_norm = tf.Variable(X_norm)
		Y_mask = (Y_norm.numpy() > 1) + (Y_norm.numpy() < -1)
		Y_norm = Y_norm.numpy()
		Y_norm[Y_mask] = 2
		Y_norm = tf.Variable(Y_norm)


	pixel_coords = tf.stack([X_norm, Y_norm], axis=2)
	return tf.reshape(pixel_coords, [b, h, w, 2])


def cam2pixel_cost(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
	"""Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: [B, 3, H, W]
        proj_c2p_rot: rotation -- b * NNN* 3 * 3
        proj_c2p_tr: translation -- b * NNN * 3 * 1
    Returns:
        array of [-1,1] coordinates -- [B, NNN, 2, H, W]
    """
	b, _, h, w = cam_coords.shape
	cam_coords_flat = tf.reshape(cam_coords, [b, 3, -1])
	pcoords = tf.matmul(proj_c2p_rot, tf.reshape(cam_coords_flat, [b, 1, 3, h * w]))

	if proj_c2p_tr is not None:
		pcoords = pcoords + proj_c2p_tr
	
	X = pcoords[:, :, 0]
	Y = pcoords[:, :, 1]
	Z = pcoords[:, :, 2]
	Z = tf.clip_by_value(Z, clip_value_min=1e-3, clip_value_max=tf.float32.max)

	X_norm = 2 * (X / Z) / (
					w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
	Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]
	if padding_mode == 'zeros':
		X_mask = (X_norm.numpy() > 1) + (X_norm.numpy() < -1)
		X_norm = X_norm.numpy()
		X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
		X_norm = tf.Variable(X_norm)

		Y_mask = (Y_norm.numpy() > 1) + (Y_norm.numpy() < -1)
		Y_norm = Y_norm.numpy()
		Y_norm[Y_mask] = 2
		Y_norm = tf.Variable(Y_norm)
	
	pixel_coords = tf.stack([X_norm, Y_norm], axis=3)
	return tf.reshape(pixel_coords, [b, -1, h, w, 2])

def cam2depth(cam_coords, proj_c2p_rot, proj_c2p_tr):
	"""Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        depth -- [B, H, W]
    """
	b, _, h, w = cam_coords.shape
	cam_coords_flat = tf.reshape(cam_coords, [b, 3, -1])
	if proj_c2p_rot is not None:
		pcoords = tf.matmul(proj_c2p_rot, cam_coords_flat)
	else:
		pcoords = cam_coords_flat
	
	if proj_c2p_tr is not None:
		pcoords = pcoords + proj_c2p_tr
	
	z = pcoords[:, 2, :]
	return tf.reshape(z, [b, h, w])


def cam2depth_cost(cam_coords, proj_c2p_rot, proj_c2p_tr):
	"""Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 3, H, W]
        proj_c2p_rot: rotation matrix of cameras -- b * nnn* 3 * 3
        proj_c2p_tr: translation vectors of cameras -- b * nnn* 3 * 1
    Returns:
        depth -- [B, nnn, H, W]
    """
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
	return tf.reshape(z, [b, n, h, w])


def depth_warp(fdepth, depth, pose, intrinsics, intrinsics_inv, padding_mode='zeros'):
	"""
    warp a target depth to the source image plane.

    Args:
        fdepth: the source depth (where to sample pixels) -- [B, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        target depth warped to the source image plane
    """
	check_sizes(depth, 'depth', 'BHW')
	check_sizes(pose, 'pose', 'B34')
	check_sizes(intrinsics, 'intrinsics', 'B33')
	check_sizes(intrinsics_inv, 'intrinsics', 'B33')
	assert(intrinsics_inv.shape == intrinsics.shape)

	batch_size, feat_height, feat_width = depth.shape

	cam_coords = pixel2cam(depth, intrinsics_inv)
	pose_mat = pose
	proj_cam_to_src_pixel = tf.matmul(intrinsics, pose_mat)
	src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:],
	                             padding_mode, rounded=True)
	projected_depth = cam2depth(cam_coords, pose_mat[:, :, :3], pose_mat[:, :, -1:])
	fdepth_expand = tf.expand_dims(fdepth, axis=1)
	fdepth_expand = tf.image.resize(fdepth_expand, [feat_height, feat_width], method='bilinear')
	warped_depth = grid_sample(fdepth_expand, src_pixel_coords, mode='nearest', padding_mode='padding_mode')
	warped_depth = tf.reshape(warped_depth, [batch_size, feat_height, feat_width])
	projected_depth = tf.clip_by_value(projected_depth, clip_value_min=1e-3, clip_value_max=tf.reduce_max(warped_depth) + 10)

	return projected_depth, warped_depth


def depth_warp_cost(fdepth, depth, pose, intrinsics, intrinsics_inv, padding_mode='zeros'):
	"""
    warp a target depth to the source image plane.

    Args:
        fdepth: the source depth (where to sample pixels) -- [B, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- b * n * n * n * 3 * 4
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        target depth warped to the source image plane
    """
	check_sizes(depth, 'depth', 'BHW')
	# check_sizes(pose, 'pose', 'BNN34')
	check_sizes(intrinsics, 'intrinsics', 'B33')
	check_sizes(intrinsics_inv, 'intrinsics', 'B33')
	assert (intrinsics_inv.shape == intrinsics.shape)

	batch_size, feat_height, feat_width = depth.shape
	pose = tf.reshape(pose, [batch_size, -1, 3, 4])  # [B,NNN, 3, 4]
	cost_n = pose.shape[1]

	cam_coords = pixel2cam(depth, intrinsics_inv)
	pose_mat = pose
	
	intrinsics = tf.reshape(intrinsics, [batch_size, 1, 3, 3])
	proj_cam_to_src_pixel = tf.matmul(intrinsics, pose_mat)
	src_pixel_coords = cam2pixel_cost(cam_coords, proj_cam_to_src_pixel[:, :, :, :3],
										proj_cam_to_src_pixel[:, :, :, -1:], padding_mode)
	src_pixel_coords = tf.reshape(src_pixel_coords, [-1, feat_height, feat_width, 2])
	projected_depth = cam2depth_cost(cam_coords, pose_mat[:, :, :, :3], pose_mat[:, :, :, -1:])
	fdepth_expand = tf.expand_dims(fdepth, axis=1)

	fdepth_expand = tf.reshape(fdepth_expand, [batch_size, 1, feat_height, feat_width])
	fdepth_expand = tf.tile(fdepth_expand, [1, cost_n, 1, 1])
	fdepth_expand = tf.reshape(fdepth_expand, [-1, 1, feat_height, feat_width])
	
	warped_depth = grid_sample(fdepth_expand, src_pixel_coords, mode='nearest',
								padding_mode=padding_mode)
	projected_depth = tf.clip_by_value(projected_depth, 1e-3, tf.reduce_max(warped_depth) + 10)
	projected_depth = tf.reshape(projected_depth, [-1, cost_n, feat_height, feat_width, 1])
	warped_depth = tf.reshape(warped_depth, [-1, cost_n, feat_height, feat_width, 1])

	return projected_depth, warped_depth	# b * nnn * h * w * 1


def inverse_warp(feat, depth, pose, intrinsics, intrinsics_inv, padding_mode='zeros'):
	"""
    Inverse warp a source image to the target image plane.

    Args:
        feat: the source feature (where to sample pixels) -- [B, CH, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
	check_sizes(depth, 'depth', 'BHW')
	check_sizes(pose, 'pose', 'B34')
	check_sizes(intrinsics, 'intrinsics', 'B33')
	check_sizes(intrinsics_inv, 'intrinsics', 'B33')

	assert (intrinsics_inv.size() == intrinsics.shape)
	batch_size, _, feat_height, feat_width = feat.shape
	
	cam_coords = pixel2cam(depth, intrinsics_inv)

	pose_mat = pose
	proj_cam_to_src_pixel =  tf.matmul(intrinsics, pose_mat)
	src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:],
								padding_mode, rounded=True)
	projected_feat = grid_sample(feat, src_pixel_coords, mode='nearest', padding_mode=padding_mode)

	return projected_feat


def inverse_warp_cost(feat, depth, pose, intrinsics, intrinsics_inv, padding_mode='zeros'):
	"""
    ref -> targets

    Args:
        feat: b * c * h * w NOTE: For tf: b * h * w * c
        depth: b * h * w
        pose: b * n (* n * n) * 3 * 4
        intrinsics: [B, 3, 3]
        intrinsics_inv: [B, 3, 3]
    """

	check_sizes(depth, 'depth', 'BHW')
	check_sizes(intrinsics, 'intrinsics', 'B33')
	check_sizes(intrinsics_inv, 'intrinsics', 'B33')

	assert (intrinsics_inv.shape == intrinsics.shape)

	batch_size, feat_height, feat_width, channel = feat.shape

	cam_coords = pixel2cam(depth, intrinsics_inv)  # [B, 3, H, W]
	pose = tf.reshape(pose, [batch_size, -1, 3, 4])  # [B,NNN, 3, 4]
	cost_n = pose.shape[1]
	
	intrinsics = tf.reshape(intrinsics, [batch_size, 1, 3, 3])
	proj_cam_to_src_pixel = tf.matmul(intrinsics, pose)

	src_pixel_coords = cam2pixel_cost(cam_coords, proj_cam_to_src_pixel[:, :, :, :3], proj_cam_to_src_pixel[:, :, :, -1:],
							padding_mode=padding_mode)
	src_pixel_coords = tf.reshape(src_pixel_coords, [-1, feat_height, feat_width, 2])
	feat = tf.reshape(tf.tile(tf.reshape(feat, [batch_size, 1, feat_height, feat_width, channel]), [1, cost_n, 1, 1, 1]), [-1, feat_height, feat_width, channel])
	projected_feat = grid_sample(feat, src_pixel_coords, mode='nearest', padding_mode=padding_mode)

	return projected_feat
