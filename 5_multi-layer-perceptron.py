# coding:utf-8

'''
Multilayer Perceptron 多层感知机。
[MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
参考 "https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py"
项目地址：https://github.com/IMLHF/tensorflow_practice

'''
from __future__ import print_function
import tensorflow as tf
import numpy as np

# 加载MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data
__mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 模型训练相关参数
__learning_rate = 0.001
__training_epochs = 10
__batch_size = 128  # 每批训练数据的大小
__display_step = 1  # 每隔__display_step周期显示一次进度

# 神经网络参数
__n_hidden_1 = 256  # 隐层第一层神经元个数
__n_hidden_2 = 256  # 隐层第二层神经元个数
__n_input = 784  # 输入层维度，与MNIST数据集的输入维度对应(图片大小28*28，展开成784维向量)
__n_output = 10  # 输出层维度，与MNIST数据集的输出维度（分类个数）对应，数字(0-9)

# tensorflow Graph输入口
__X_input = tf.placeholder("float", [None, __n_input])
__Y_true = tf.placeholder("float", [None, __n_output])

# 定义网络中的权重和阈值（偏置）
__weights = {
    'w_hidden_layer1': tf.Variable(tf.random_normal([__n_input, __n_hidden_1])),
    'w_hidden_layer2': tf.Variable(tf.random_normal([__n_hidden_1, __n_hidden_2])),
    'w_out': tf.Variable(tf.random_normal([__n_hidden_2, __n_output])),
}
__biases = {
    'b_hidden_layer1': tf.Variable(tf.random_normal([__n_hidden_1])),
    'b_hidden_layer2': tf.Variable(tf.random_normal([__n_hidden_2])),
    'b_out': tf.Variable(tf.random_normal([__n_output])),
}


# 创建多层感知机模型（双隐层）
def multilayer_perceptron(__x_input_t):
  __hidden_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(__x_input_t, __weights['w_hidden_layer1']),
                                         __biases['b_hidden_layer1']))
  __hidden_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(__hidden_layer1, __weights['w_hidden_layer2']),
                                         __biases['b_hidden_layer2']))
  ___out_layer = tf.nn.sigmoid(tf.add(tf.matmul(__hidden_layer2, __weights['w_out']),
                                      __biases['b_out']))
  return ___out_layer


# 训练和测试
if __name__ == '__main__':
  _logits = multilayer_perceptron(__X_input)
  # 使用softmax建立回归模型
  # # 使用交叉熵作为损失函数
  '''三种不同的方法求交叉熵'''
  # __loss_cross_entropy = - \
  #     tf.reduce_mean(__Y_true*tf.nn.log_softmax(_logits))  # 准确率在95%左右
  __loss_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits=_logits, labels=__Y_true)) # 准确率在95%左右
  # __loss_cross_entropy = -tf.reduce_mean(__Y_true*tf.log(tf.nn.softmax(_logits))) # 准确率在75%左右

  # 使用Adam算法优化模型
  __train_op = tf.train.AdamOptimizer(
      learning_rate=__learning_rate).minimize(__loss_cross_entropy)
  # 初始化变量
  __init = tf.global_variables_initializer()
  # 开始训练并测试
  with tf.Session() as __session_t:
    __session_t.run(__init)

    for epoch in range(__training_epochs):
      __avg_lost = 0
      __total_batch = int(__mnist.train.num_examples/__batch_size)
      for i in range(__total_batch):
        __x_batch, __y_batch = __mnist.train.next_batch(__batch_size)
        # print(__y_batch[0])
        __nouse, __loss_t = __session_t.run([__train_op, __loss_cross_entropy],
                                            feed_dict={__X_input: __x_batch,
                                                       __Y_true: __y_batch})
        __avg_lost += float(__loss_t)/__total_batch
        # region debug
        # print(__loss_t)
        # tmp = __session_t.run(
        #     _logits, feed_dict={__X_input: __x_batch, __Y_true: __y_batch})
        # print(np.shape(_logits))
        # for tt in tmp:
        #   for pp in tt:
        #     print("%.2f, " % pp, end="")
        #   print()
        # print(__y_batch)
        # endregion edbug
      if epoch % __display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "Avg_Loss=", __avg_lost)
    print("Optimizer Finished!")

    # 测试模型
    __predict = tf.nn.softmax(multilayer_perceptron(__X_input))
    __correct = tf.equal(tf.argmax(__predict, 1), tf.argmax(__Y_true, 1))
    __accuracy_rate = tf.reduce_mean(tf.cast(__correct, tf.float32))
    print("Accuracy:", __session_t.run(__accuracy_rate,
                                       feed_dict={__X_input: __mnist.test.images,
                                                  __Y_true: __mnist.test.labels}))
