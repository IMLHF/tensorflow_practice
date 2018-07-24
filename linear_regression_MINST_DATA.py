# coding:utf-8
# 参考 "https://blog.csdn.net/u013569304/article/details/81175006"
#      "http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html"
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

# 数据集位置，如果该位置没有数据会自动下载，不过可能因为网络问题无法下载。
# 可以从http://yann.lecun.com/exdb/mnist/下载，一共四个文件，下载完不用解压。
__file = 'MNIST_data/'
__mnist = input_data.read_data_sets(__file, one_hot=True)

# 训练数据的格式，行数为None，表示不固定；
# 列数为784，是MNIST数据集输入数维度，输入集是28*28的点阵，展开后就是一个784维的向量
# 输出集是十个不同的类别（0-9），使用one_hot类型的输出集
__input_x = tf.placeholder(tf.float32, [None, 784])
__train_y_true = tf.placeholder(tf.float32, [None, 10])

# 模型的权重和偏移量
# 变量__W和__b是线性模型的参数
# __W是一个784∗10的矩阵，因为输入有784个特征，同时有10个输出值。
# __b是一个10维的向量，是因为输出有10个分类。
__W = tf.Variable(tf.zeros([784, 10]))
__b = tf.Variable(tf.zeros([10]))

# 创建Session
__session = tf.InteractiveSession()

# 初始化权重变量
__session.run(tf.global_variables_initializer())

# 构建回归模型
__output_y = tf.nn.softmax(tf.matmul(__input_x, __W) + __b)

# 交叉熵（损失函数）
__cross_entropy = -tf.reduce_sum(__train_y_true*tf.log(__output_y))

# region 训练
# 使用梯度下降法优化模型，学习率为0.01。
# 训练周期为1000，每个训练周期使用60个样本
__train_session = tf.train.GradientDescentOptimizer(0.01
                                                    ).minimize(__cross_entropy)
for i in range(1000):
  __batch = __mnist.train.next_batch(60)
  __train_session.run(
      feed_dict={__input_x: __batch[0], __train_y_true: __batch[1]})
# endregion


# region 测试
# 从测试集取200个样本测试
__test_x, __test_y_true = __mnist.test.next_batch(200)

# 使用训练好的模型预测测试集结果（得到一个list）
__perdict_num_list = __session.run(
    tf.argmax(__output_y, 1), feed_dict={__input_x: __test_x})
    
__accuracy = 0
for i in range(len(__test_x)):
  __test_y_true_i = np.argmax(__test_y_true[i])
  # session.run(tf.argmax(__test_y[i],1)) ????
  print("Test", i, ",[ Perdiction Value:", __perdict_num_list[i],
        "  True Value:", __test_y_true_i, "]", end="")
  if __perdict_num_list[i] == __test_y_true_i:
    print(', Answer: pass')
    __accuracy += 1./len(__test_x)
  else:
    print(', Answer: fail')
print("Accuracy:", __accuracy)
# endregion

__session.close()
