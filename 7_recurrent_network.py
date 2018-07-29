from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
__mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 训练参数
__learning_rate = 0.001
__training_steps = 20000
__batch_size = 128
__display_step = 100

# Network Parameters
__num_input = 28  # 输入
__timesteps = 28  # 时间轴
__num_hidden = 128  # 隐层数量
__num_classes = 10  # 分类数量

# tf Graph input
__X_input = tf.placeholder("float", [None, __timesteps, __num_input])
__Y_true = tf.placeholder("float", [None, __num_classes])

# Define weights
__weights = {
    'out': tf.Variable(tf.random_normal([__num_hidden, __num_classes]))
}
__biases = {
    'out': tf.Variable(tf.random_normal([__num_classes]))
}


def RNN(__x_t, __weights, __biases):

  # 准备RNN输入数据，输入数据shape为[batch_size, timesteps, n_input]
  # 需要数据为__timesteps个shape为[batch_size, n_input]的tensor
  # 使用unstack拆分tensor，第一个参数为要拆分的tensor，
  # 第二个参数num，表示拆分为num个tensor
  # 第三个参数axis表示要拆分的维度
  __x_t = tf.unstack(__x_t, __timesteps, axis=1)

  # 定义一个LSTM单元
  __lstm_cell = rnn.BasicLSTMCell(__num_hidden, forget_bias=1.0)

  # 获取lstm单元输出
  __outputs, __states = rnn.static_rnn(__lstm_cell, __x_t, dtype=tf.float32)

  # Linear activation, using rnn inner loop last output
  return tf.matmul(__outputs[-1], __weights['out']) + __biases['out']


__logits = RNN(__X_input, __weights, __biases)
__prediction = tf.nn.softmax(__logits)

# 确定损失函数，定义训练节点
__loss_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=__logits, labels=__Y_true))
__optimizer = tf.train.GradientDescentOptimizer(learning_rate=__learning_rate)
__train_op = __optimizer.minimize(__loss_cross_entropy)

# 评估模型
__correct_pred = tf.equal(tf.argmax(__prediction, 1), tf.argmax(__Y_true, 1))
__accuracy = tf.reduce_mean(tf.cast(__correct_pred, tf.float32))

__init = tf.global_variables_initializer()

# 训练
with tf.Session() as __sess_t:

  __sess_t.run(__init)

  for __step in range(1, __training_steps+1):
    __batch_x, __batch_y = __mnist.train.next_batch(__batch_size)
    # 改变数据形状
    __batch_x = __batch_x.reshape((__batch_size, __timesteps, __num_input))
    # Run optimization op (backprop)
    __sess_t.run(__train_op, feed_dict={
                 __X_input: __batch_x, __Y_true: __batch_y})
    if __step % __display_step == 0 or __step == 1:
      # Calculate batch loss and accuracy
      __loss, __acc = __sess_t.run(
          [__loss_cross_entropy, __accuracy], feed_dict={__X_input: __batch_x,
                                                         __Y_true: __batch_y})
      print("Step " + str(__step) + ", Minibatch Loss= " +
            "{:.4f}".format(__loss) + ", Training Accuracy= " +
            "{:.3f}".format(__acc))

  print("Optimization Finished!")

  # 正确率
  __test_len = 6000
  __test_data = __mnist.test.images[:__test_len].reshape(
      (-1, __timesteps, __num_input))
  __test_label = __mnist.test.labels[:__test_len]
  print("Testing Accuracy:",
        __sess_t.run(__accuracy, feed_dict={__X_input: __test_data, __Y_true: __test_label}))
