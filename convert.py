import numpy as np
import tensorflow as tf		# tf 2.3



# no depth2normal
# no imgrad
# no imgrad_yx
# no b_inv
# no GradLoss


def matrix2angle(matrix):
	"""
    ref: https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py
    input size: ... * 3 * 3, "matrix" is a tf Tensor object
    output size:  ... * 3, output is also a tf Tensor object
    """
	i = 0
	j = 1
	k = 2
	
	M = tf.cast(tf.reshape(matrix, (-1, 3, 3)), float)

	cy = tf.sqrt(M[:, i, i] * M[:, i, i] + M[:, j, i] * M[:, j, i])

	if tf.reduce_max(cy) > 1e-15 * 4:
		ax = tf.atan2(M[:, k, j], M[:, k, k])
		ay = tf.atan2(-M[:, k, i], cy)
		az = tf.atan2(M[:, j, i], M[:, i, i])
	else:
		ax = tf.atan2(-M[:, j, k], M[:, j, j])
		ay = tf.atan2(-M[:, k, i], cy)
		az = tf.zero(matrix.shape[:-1])
	return tf.concat([tf.expand_dims(ax, -1), tf.expand_dims(ay, -1), tf.expand_dims(az, -1)], -1)


def angle2matrix(angle):
	"""
    ref: https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py
    input size:  ... * 3  -> tf.Tensor object
    output size: ... * 3 * 3 -> tf.Tensor object
    """
	dims = [dim for dim in angle.shape]
	angle = tf.reshape(angle, (-1, 3))

	i = 0
	j = 1
	k = 2
	ai = angle[:, 0]
	aj = angle[:, 1]
	ak = angle[:, 2]
	si, sj, sk = tf.sin(ai), tf.sin(aj), tf.sin(ak)
	ci, cj, ck = tf.cos(ai), tf.cos(aj), tf.cos(ak)
	cc, cs = ci * ck, ci * sk
	sc, ss = si * ck, si * sk

	M = tf.eye(3)
	M = tf.reshape(M, (1, 3, 3))
	M = tf.tile(M, [angle.shape[0], 1, 1])
	M = M.numpy()

	M[:, i, i] = cj * ck
	M[:, i, j] = sj * sc - cs
	M[:, i, k] = sj * cc + ss
	M[:, j, i] = cj * sk
	M[:, j, j] = sj * ss + cc
	M[:, j, k] = sj * cs - sc
	M[:, k, i] = -sj
	M[:, k, j] = cj * si
	M[:, k, k] = cj * ci

	return tf.reshape(tf.Variable(M), dims + [3])

def inv(A, eps=1e-10):
	# "invert" a M X N X N tensor
	# input -- M X N X N tensor
	# output -- M X N X N tensor

	# note, it is mathematically nonsense to invert a rank 3 tensor. The origional function simply inverts every M matrix of size N X N and stacks them 

	assert len(A.shape) == 3 and \
	       A.shape[1] == A.shape[2]
	
	A = A.numpy()
	A_mats = np.split(A, A.shape[0])
	A_inv = np.linalg.inv(A_mats).reshape(A.shape)

	return A_inv
