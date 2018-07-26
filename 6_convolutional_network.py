# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

__mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
__session = tf.InteractiveSession()


'''
conv2d              卷积
@:param x           输入图像
@:param w           卷积核
@:return Tensor     做过卷积后的图像
'''

'''
tf.nn.conv2d 卷积函数
参数 input 输入图像              四维[batch, height, width, channels]
参数 filter 卷积核              四维[batch, height, width, channels]
参数 strides 卷积核移动步长     四维[batch, height, width, channels]
参数 padding 边缘处理方式       SAME和VALID,SAME就是可以在外围补0再卷积，VALID会检查步长是否合理，不能补0
返回 Tensor
'''


def conv2d(__x_input, __filter):
  return tf.nn.conv2d(__x_input, __filter, strides=[1, 1, 1, 1], padding='SAME')


'''
tf.nn.max_pool 卷积函数
参数 value 输入图像               四维[batch, height, width, channels]
参数 ksize 池化窗口              四维[batch, height, width, channels]
参数 strides 池化窗口移动步长    四维[batch, height, width, channels]，
一般不对batch和图像通道数进行池化，所以ksize和strides的batch个channels都是1
参数 padding 边缘处理方式      SAME和VALID,SAME就是可以在外围补0再卷积，VALID会检查步长是否合理，不能补0
返回 Tensor
'''


def max_pooling_2x2(__max_pooling_target):
  return tf.nn.max_pool(__max_pooling_target, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


__X_input = tf.placeholder(tf.float32, [None, 784])
__Y_true = tf.placeholder(tf.float32, [None, 10])

'''
tf.reshape 重定形状
参数 tensor 输入数据
参数 shape 形状                按此shape生成相应数组，但-1是特例，表示有此维度，但是数值不定
'''
__x_image = tf.reshape(__X_input, [-1, 28, 28, 1])

# 构建神经网络
# 截断正态分布获取随机值
__w_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1))
__b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
# 激活函数
__h_conv1 = tf.nn.relu(conv2d(__x_image, __w_conv1) + __b_conv1)
# 池化
__h_pool1 = max_pooling_2x2(__h_conv1)
''' yiceng'''
__w_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1))
__b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
__h_conv2 = tf.nn.relu(conv2d(__h_pool1, __w_conv2) + __b_conv2)
__h_pool2 = max_pooling_2x2(__h_conv2)

__w_fc1 = tf.Variable(tf.truncated_normal(
    shape=[7 * 7 * 64, 1024], stddev=0.1))
__b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
__h_pool2_flat = tf.reshape(__h_pool2, [-1, 7 * 7 * 64])
__h_fc1 = tf.nn.relu(tf.matmul(__h_pool2_flat, __w_fc1) + __b_fc1)

__keep_probability = tf.placeholder(tf.float32)
############################################################################################
# tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)   防止过拟合   在对输入的数据进行一定的取舍，从而降低过拟合
# 参数 x 输入数据
# 参数 keep_prob 保留率             对输入数据保留完整返回的概率
# 返回 Tensor
############################################################################################
__h_fc1_drop = tf.nn.dropout(__h_fc1, __keep_probability)

__w_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.1))
__b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

# tf.nn.softmax(logits, dim=-1, name=None): SoftMax函数
# 参数 logits 输入            一般输入是logit函数的结果
# 参数 dim 卷积核             指定是第几个维度，默认是-1，表示最后一个维度
# 返回 Tensor
__y_conv = tf.nn.softmax(tf.matmul(__h_fc1_drop, __w_fc2) + __b_fc2)


cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(__Y_true * tf.log(__y_conv), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(__y_conv, 1), tf.argmax(__Y_true, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()  # 启动Session
for i in range(2000):
  # Datasets.train.next_batch 批量处理记录数
  # 返回 [image,label]
  __x_batch,__y_batch = __mnist.train.next_batch(50)
  if i % 100 == 0:
    train_accuracy = accuracy.eval(
        feed_dict={__X_input: __x_batch, __Y_true: __y_batch, __keep_probability: 0.99})
    print("step %d, training accuracy %g" % (i, train_accuracy))

  train_step.run(feed_dict={__X_input: __x_batch, __Y_true: __y_batch, __keep_probability: 0.5})

print("test accuracy %g" % accuracy.eval(
    feed_dict={__X_input: __mnist.test.images, __Y_true: __mnist.test.labels, __keep_probability: 1.0}))
