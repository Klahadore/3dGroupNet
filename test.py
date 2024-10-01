import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from groupy.gconv.tensorflow_gconv.splitgconv3d import gconv3d, gconv3d_util
import groupy
import os
path = os.path.abspath(groupy.__file__)

# Construct graph
x = tf.placeholder(tf.float32, [None, 9, 9, 9, 3])

# Initialize gconv3d utility for the first convolution
gconv_indices, gconv_shape_info, w_shape = gconv3d_util(
    h_input='Z3', h_output='OH', in_channels=3, out_channels=64, ksize=3)
w = tf.Variable(tf.truncated_normal(w_shape, stddev=1.))
y = gconv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME',
            gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)

# Initialize gconv3d utility for the second convolution
gconv_indices, gconv_shape_info, w_shape = gconv3d_util(
    h_input='OH', h_output='OH', in_channels=64, out_channels=64, ksize=3)
w = tf.Variable(tf.truncated_normal(w_shape, stddev=1.))
y = gconv3d(y, w, strides=[1, 1, 1, 1, 1], padding='SAME',
            gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)

# Compute
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# Correct the shape of the input data fed to the placeholder
y_output = sess.run(y, feed_dict={x: np.random.randn(10, 9, 9, 9, 3)})
sess.close()

print(y_output.shape)  # Expected output shape: (10, 9, 9, 9, 64)
