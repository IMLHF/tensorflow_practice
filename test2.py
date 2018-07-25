import tensorflow as tf

# our NN's output
logits = tf.constant([[-2438.34, 1612.02, -1284.87, 1238.97, 1742.21, 989.41, 2646.55, 886.95, -1538.45, -501.97],
                      [-4292.72, -2750.94, -406.36, 37.57, 1434.28, -2815.56, 5310.22, -1549.30, -2699.31, 2641.84],
                      [-2056.02, -3174.13, 33.34, -3564.92, 659.35, -1136.72, 1984.09, -2577.76, -2321.90, 1964.23]])
# step1:do softmax
max_t=tf.reduce_max(logits,1)
logits2 = tf.transpose(tf.transpose(logits)-max_t)
y = tf.nn.softmax(logits2)
# true label
y_ = tf.constant([[0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
# step2:do cross_entropy
cross_entropy = -tf.reduce_mean(y_*tf.nn.log_softmax(logits))
# cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-29,1.0)))
# do cross_entropy just one step
cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=y_))  # dont forget tf.reduce_sum()!!

with tf.Session() as sess:
  softmax = sess.run(y)
  c_e = sess.run(cross_entropy)
  c_e2 = sess.run(cross_entropy2)
  max_tt=sess.run(logits2)
  print(max_tt)
  print("step1:softmax result=")
  print(softmax)
  print("step2:cross_entropy result=")
  print(c_e)
  print("Function(softmax_cross_entropy_with_logits) result=")
  print(c_e2)
