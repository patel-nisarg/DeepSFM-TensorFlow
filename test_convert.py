import unittest
from convert import matrix2angle, angle2matrix, inv
import tensorflow as tf
import numpy as np


class Mat2EulerTest(unittest.TestCase):

    # test mat2euler with 1 rot_matrix (3, 3)
    def test_mat2euler_one_mat(self):
        mat = tf.constant([[1, 2, 3], [4, 2, 2], [5, 6, 3]])
        ang_numpy = np.array([[ 1.1071, -0.8812,  1.3258]])
        self.assertIsNone(np.testing.assert_array_almost_equal(matrix2angle(mat).numpy(), ang_numpy, decimal=3))

    # test mat2euler with 1 rot_matrix (1, 3, 3)
    def test_mat2euler_extra_dim(self):
        mat = tf.constant([[[1, 2, 3], [4, 2, 2], [5, 6, 3]]])
        ang_numpy = np.array([[ 1.1071, -0.8812,  1.3258]])
        self.assertIsNone(np.testing.assert_array_almost_equal(matrix2angle(mat).numpy(), ang_numpy, decimal=3))

    # test mat2euler with multiple rot_matrices (2, 3, 3)
    def test_mat2euler_three_mat(self):
        mat = tf.constant([[[1, 2, 3], [4, 2, 2], [5, 6, 3]], 
        [[8, 2, 3], [4, 6, 2], [5, 7, 3]]])
        ang_numpy = np.array([[ 1.1071, -0.8812,  1.3258],
        [ 1.1659, -0.5097,  0.4636]])
        self.assertIsNone(np.testing.assert_array_almost_equal(matrix2angle(mat).numpy(), ang_numpy, decimal=3))

    # test angle2matrix using multiple euler angles
    def test_angle2matrix(self):
        angle = tf.constant([[0.2342, 1.23423, -1.23423], [2.23423, -2.3234, 0.5433]])
        mat_numpy = np.array([[[ 0.1091,  0.9905,  0.0842],
         [-0.3117,  0.1145, -0.9433],
         [-0.9439,  0.0766,  0.3212]],

        [[-0.5851, -0.1739,  0.7921],
         [-0.3534, -0.8244, -0.4421],
         [ 0.7299, -0.5386,  0.4209]]])
        self.assertIsNone(np.testing.assert_array_almost_equal(angle2matrix(angle).numpy(), mat_numpy, decimal=3))

    # test inverse tensor function
    def test_inverse(self):
        A = tf.constant([[[3, 2], [2, 4]], [[6, 6], [7, 5]]])
        A_inv = np.array([[[ 0.5000, -0.2500],
         [-0.2500,  0.3750]],
        [[-0.4167,  0.5000],
         [ 0.5833, -0.5000]]])
        self.assertIsNone(np.testing.assert_array_almost_equal(inv(A), A_inv, decimal=3))


if __name__ == '__main__':
    unittest.main()

