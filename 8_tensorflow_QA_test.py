import tensorflow as tf
#tensorflow question.pdf的部分测试代码

def mat_mul():
  A = tf.random_normal([2, 1, 4, 5])
  B = tf.random_normal([2, 3, 1, 5])
  return tf.shape(tf.multiply(A, B))


def where():
  A = tf.random_normal([4, 3])
  B = tf.random_normal([4, 3])
  return A, B, tf.where(A > B, A, B)


def while_loop():
  A = tf.random_normal([2, 3, 4])

  def cond(n, i, other_param):
    return i < n

  def body(n, i, other_param):
    i += 1
    other_param = other_param+"???"
    return (n, i, other_param)

  # return tf.while_loop(cond, body, (tf.shape(A)[2], 0, ''))
  return tf.while_loop(lambda n, i, other_param: i < n,
                       lambda n, i, other_param: (n, i+1, other_param+'???'),
                       (tf.shape(A)[2], 0, ''))


def condd():
  A = tf.random_normal([2, 3, 10])
  return tf.cond(tf.equal(tf.shape(A)[2],10), lambda: 'yesyes', lambda: 'nono')

def tf_scope_test():
  with tf.name_scope('namescope'):
    var_1 = tf.Variable(initial_value=[0], name='var_1')
    var_2 = tf.get_variable(name='var_2', shape=[1, ])
  with tf.variable_scope('variable_scope'):
    var_3 = tf.Variable(initial_value=[0], name='var_3')
    var_4 = tf.get_variable(name='var_4', shape=[1, ])

  print(var_1.name)
  print(var_2.name)
  print(var_3.name)
  print(var_4.name)

def tf_add_node_mem_test():  # 新建op节点
  ini = tf.constant_initializer(1, dtype=tf.int32)
  w = tf.get_variable("test", [1],
                      initializer=ini,
                      dtype=tf.int32)
  i = 0
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x = tf.placeholder(tf.int32)
    while True:
      op = tf.multiply(w, x)
      tmp = sess.run(op, feed_dict={x: i})
      i += 1
      if i % 200 == 0:
        print(tmp)

def tf_reuse_node_mem_test():  # 复用节点
  ini = tf.constant_initializer(1, dtype=tf.int32)
  w = tf.get_variable("test", [1],
                      initializer=ini,
                      dtype=tf.int32)
  i = 0
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x = tf.placeholder(tf.int32)
    op = tf.multiply(w, x)
    while True:
      tmp = sess.run(op, feed_dict={x: i})
      i += 1
      if i % 200 == 0:
        print(tmp)

def concat():
  A=tf.random_normal([2,3])
  B=tf.random_normal([2,5])
  return tf.shape(tf.concat([A,B],axis=-1))

if __name__ == "__main__":
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # tmp = sess.run(mat_mul())
    # tmp = sess.run(where())
    tmp = sess.run(while_loop())
    # tmp = sess.run(condd())
    # tf_scope_test()
    # tmp=sess.run(concat())
    print(tmp)
