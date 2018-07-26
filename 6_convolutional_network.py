# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

__mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

__learning_rate=0.0001
__training_epochs = 600
__batch_size = 100  # 每批训练数据的大小
__display_step = 10  # 每隔__display_step批次显示一次进度

'''
tf.nn.conv2d 卷积函数
参数 input 输入图像              四维，shape如[batch, in_height, in_width, in_channels]
参数 filter 卷积核               四维，shape如[filter_height, filter_width, in_channels, out_channels]
参数 strides 卷积核移动步长       列表，卷积时在图像每一维的步长
参数 padding 边缘处理方式       SAME和VALID,SAME就是可以在外围补0再卷积，VALID会检查步长是否合理，不能补0
返回 Tensor
'''
def conv2d(__x_input, __conv_kernel):
  return tf.nn.conv2d(__x_input, __conv_kernel, strides=[1, 1, 1, 1], padding='SAME')


'''
tf.nn.max_pool 卷积函数
参数 value 输入图像               四维，shape如[batch_num, height, width, channels]
参数 ksize 池化窗口大小            列表[batch, height, width, channels]
参数 strides 池化窗口移动步长       列表[batch, height, width, channels]，
一般不对batch和图像通道数进行池化，所以ksize和strides的batch个channels都是1
参数 padding 边缘处理方式      SAME和VALID,SAME就是可以在外围补0再卷积，VALID会检查步长是否合理，不能补0
返回 Tensor
'''
def max_pooling_2x2(__max_pooling_target):
  return tf.nn.max_pool(__max_pooling_target, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



#定义权值和偏置
__weights = {
    # 截断正态分布获取随机值
    # [batch, in_height, in_width, in_channels]
    # 5x5x1x32,卷积核的视野大小是5x5，输入图像通道为1，该层卷积核32个filter
    '__w_conv1': tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1)),
    # [filter_height, filter_width, in_channels, out_channels]
    # 5x5x32x64,同样是卷积核的视野大小是5x5，因为上一层使用了32个filter，
    # 所以上层生成结果的通道为32，该层卷积核有32个filter
    '__w_conv2': tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1)),
    '__w_fc1': tf.Variable(tf.truncated_normal(shape=[7 * 7 * 64, 1024], stddev=0.1)),
    '__w_fc2': tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.1)),
}
__biases={
    '__b_conv1': tf.Variable(tf.constant(0.1, shape=[32])),
    '__b_conv2': tf.Variable(tf.constant(0.1, shape=[64])),
    '__b_fc1': tf.Variable(tf.constant(0.1, shape=[1024])),
    '__b_fc2': tf.Variable(tf.constant(0.1, shape=[10]))
}

# region 创建CNN结构
def conv_net(__x_t,__keep_probability_t):
  #shape，-1表示有此维度，但是数值不定
  __x_image = tf.reshape(__x_t, [-1, 28, 28, 1])
  __h_conv1 = tf.nn.relu(conv2d(__x_image, __weights['__w_conv1']) + __biases['__b_conv1'])
  __h_pool1 = max_pooling_2x2(__h_conv1)
  __h_conv2 = tf.nn.relu(conv2d(__h_pool1, __weights['__w_conv2']) + __biases['__b_conv2'])
  __h_pool2 = max_pooling_2x2(__h_conv2)
  __h_pool2_flat = tf.reshape(__h_pool2, [-1, 7 * 7 * 64])
  __h_fc1 = tf.nn.relu(tf.matmul(__h_pool2_flat, __weights['__w_fc1']) + __biases['__b_fc1'])

  # tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
  # 防止过拟合   在对输入的数据进行一定的取舍，从而降低过拟合
  # 参数 x 输入数据
  # 参数 keep_prob 保留率             对输入数据保留完整返回的概率
  # 返回 Tensor
  __h_fc1_drop = tf.nn.dropout(__h_fc1, __keep_probability_t)

  # tf.nn.softmax(logits, dim=-1, name=None): SoftMax函数
  # 参数 logits 输入            一般输入是logit函数的结果
  # 参数 dim 卷积核             指定是第几个维度，默认是-1，表示最后一个维度
  # 返回 Tensor
  __y_conv_logits = tf.matmul(__h_fc1_drop, __weights['__w_fc2']) + __biases['__b_fc2']
  return __y_conv_logits
#endregion


__X_input = tf.placeholder(tf.float32, [None, 784])
__Y_true = tf.placeholder(tf.float32, [None, 10])
__keep_probability = tf.placeholder(tf.float32)

__logits=conv_net(__X_input,__keep_probability)
__out_softmax=tf.nn.softmax(__logits)
# __loss_cross_entropy = tf.reduce_mean(
#     -tf.reduce_sum(__Y_true * tf.log(__out_softmax), axis=1))
__loss_cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=__logits, labels=__Y_true))

__train_op = tf.train.AdamOptimizer(__learning_rate).minimize(__loss_cross_entropy)

__accuracy = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(__out_softmax, 1), tf.argmax(__Y_true, 1)), tf.float32))

init=tf.global_variables_initializer()

with tf.Session() as __session_t:
  __session_t.run(init)
  for i in range(__training_epochs):
    # Datasets.train.next_batch 获取批量处样本
    # 返回 [image,label]
    __x_batch,__y_batch = __mnist.train.next_batch(__batch_size)

    if i % __display_step == 0:
      __train_accuracy = __session_t.run(
          __accuracy,
          feed_dict={__X_input: __x_batch,
                     __Y_true: __y_batch,
                     __keep_probability: 1.0})
      print("step %d, training accuracy %g" % (i, __train_accuracy))

    __train_op.run(feed_dict={__X_input: __x_batch, __Y_true: __y_batch, __keep_probability: 0.5})

  print("test accuracy %g" % __session_t.run(
      __accuracy,
      feed_dict={__X_input: __mnist.test.images,
                 __Y_true: __mnist.test.labels,
                 __keep_probability: 1.0}))
