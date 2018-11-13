import tensorflow as tf
import numpy as np

# # channel
# x = tf.Variable(np.random.rand(10, 257, 9,3), dtype=np.float32)
# kernel = tf.Variable(np.random.rand(257,7,2,55), dtype=np.float32)
# y = tf.nn.conv2d(x, kernel, strides=[1,1,2,1], padding='VALID')
# # channel==inputchannel strides[3]无效
# print(y)
# channel
# x = tf.Variable(np.random.rand(1, 8, 257), dtype=np.float32)
# kernel = tf.Variable(np.random.rand(7,257,1), dtype=np.float32)
# y = tf.nn.conv1d(x, kernel, stride=1, padding='SAME')
# print(y)
# x = tf.Variable(np.random.rand(10, 2, 9,257), dtype=np.float32)
# kernel = tf.Variable(np.random.rand(1,7,256,1), dtype=np.float32)
# y = tf.nn.conv2d(x, kernel, strides=[1,1,2,1], padding='VALID')
# print(y)
x=np.reshape([1,2,3],[3,1])
y=np.reshape([1,2],[1,2])
print(x+y)
t=tf.placeholder("float",shape=[2,3])
print(np.shape(t))
dc={
  1:2,
  3:4
}
print(dc[1])
