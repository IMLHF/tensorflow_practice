import tensorflow as tf
import numpy as np
import os
import shutil

decode=False
resume_training=False

save_dir='_'+__file__[:__file__.rfind('.')]

if not decode:
  if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
  os.makedirs(save_dir)

graph=tf.Graph()
with graph.as_default():
  with tf.variable_scope('variable'):
    w=tf.get_variable(name='w',initializer=tf.constant([0]),dtype=tf.int32)
    rand_inc=tf.random_uniform(minval=1,maxval=3,dtype=tf.int32,shape=[1])
    update_w=tf.assign_add(w,rand_inc)
  saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30) # 变量创建之后

  if decode:
    with tf.Session(graph=graph) as sess:
      ckpt = tf.train.get_checkpoint_state(save_dir)
      saver.restore(sess,ckpt.model_checkpoint_path)
      print(sess.run(w))

  else:
    n_epoch=10
    with tf.Session(graph=graph) as sess:
      if resume_training:
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess,ckpt.model_checkpoint_path)
        else:
          print('Error,checkpoint not exist,can\'t resume training.')
      else:
        sess.run(tf.global_variables_initializer())
        saver.save(sess,os.path.join(save_dir,'initial_model'))

      for i_epoch in range(n_epoch):
        w_forward=sess.run(w)
        sess.run(update_w)
        w_backward=sess.run(w)
        print(w_forward,w_backward)
        if w_backward-w_forward>1:
          '''
          ckpt.model_checkpoint_path返回最新的模型检查点，
          但是不是动态返回的，所以每次找最新检查点时使用
          tf.train.get_checkpoint_state(save_dir)新建ckpt对象
          '''
          ckpt = tf.train.get_checkpoint_state(save_dir)
          if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            print("Training rejected.")
          else:
            print('Error,checkpoint not exist,can\'t reject training.')
        else:
          saver.save(sess,os.path.join(save_dir,'epoch_%02d' % (i_epoch+1)))
          print("Traing accepted.")
        print('#########################')

