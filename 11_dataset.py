import tensorflow as tf
import numpy as np


def tf_normalize_spec(spec):
  upbound = 7.0
  lowbound = -1.0
  spec = tf.divide(tf.log(spec+0.5), tf.log(10.0))
  spec = tf.clip_by_value(spec, lowbound, upbound)
  spec = tf.subtract(spec, lowbound)
  spec = tf.divide(spec, tf.subtract(upbound, lowbound))
  return spec


def tf_rmNormalization_spec(spec):
  upbound = 7.0
  lowbound = -1.0
  spec = tf.math.pow(10.0, spec)-0.5
  spec = tf.where(spec > 0.0, spec, 0.0)
  spec = tf.multiply(spec*(upbound-lowbound))+lowbound
  return spec


def extract_X_Y_Lengths(example_proto):
  # return example_proto,tf.random_uniform(shape=[]),233,233
  features = {
      'mat': tf.FixedLenFeature([4, 3, 160000], tf.float32),
      'shape': tf.FixedLenFeature([3], tf.int64),
  }
  dataset = tf.parse_single_example(example_proto, features)
  # shape=dataset['shape']
  mat = tf.cast(dataset['mat'], tf.float32)
  speaker_num = tf.shape(mat)[0]
  speaker1id = tf.random_uniform([], maxval=speaker_num, dtype=tf.int32)
  speaker2id = tf.random_uniform([], maxval=speaker_num, dtype=tf.int32)
  speaker1id, speaker2id = tf.while_loop(lambda id1, id2: tf.equal(id1, id2),
                                         lambda id1, id2: (tf.random_uniform(
                                             [], maxval=speaker_num, dtype=tf.int32),
                                             tf.random_uniform(
                                             [], maxval=speaker_num, dtype=tf.int32)),
                                         (speaker1id, speaker2id))
  utt1id = tf.random_uniform([], maxval=tf.shape(mat)[1], dtype=tf.int32)
  utt2id = tf.random_uniform([], maxval=tf.shape(mat)[1], dtype=tf.int32)
  waveData1 = mat[speaker1id][utt1id]
  waveData2 = mat[speaker2id][utt2id]
  waveData1 = tf.multiply(
      tf.divide(waveData1, tf.reduce_max(tf.abs(waveData1))), 32767.0)
  waveData2 = tf.multiply(
      tf.divide(waveData2, tf.reduce_max(tf.abs(waveData2))), 32767.0)

  mixed_wave = tf.divide(tf.cast(tf.add(waveData1, waveData2), tf.float32), 2)
  mixed_spec = tf.abs(tf.contrib.signal.stft(signals=mixed_wave,
                                             frame_length=512,
                                             frame_step=256,
                                             fft_length=512))
  wave1_spec = tf.abs(tf.contrib.signal.stft(signals=waveData1,
                                             frame_length=512,
                                             frame_step=256,
                                             fft_length=512))
  wave2_spec = tf.abs(tf.contrib.signal.stft(signals=waveData2,
                                             frame_length=512,
                                             frame_step=256,
                                             fft_length=512))
  mixed_spec = tf_normalize_spec(mixed_spec)
  wave1_spec = tf_normalize_spec(wave1_spec)
  wave2_spec = tf_normalize_spec(wave2_spec)
  return mixed_spec, wave1_spec, wave2_spec, tf.shape(wave1_spec)[0]


if __name__ == '__main__':
  train_tfrecord = [
      '/mnt/d/tf_recipe/ALL_DATA/aishell/mixed_data_small.tfrecords']
  train_set = tf.data.TFRecordDataset(train_tfrecord)
  # train_set=tf.data.Dataset.range(1)
  train_set = train_set.repeat(4)
  # train_set = train_set.map(
  #     map_func=extract_X_Y_Lengths, num_parallel_calls=FLAGS.num_threads_processing_data)
  # train_set = train_set.batch(FLAGS.batch_size)
  train_set = train_set.apply(tf.contrib.data.map_and_batch(
      map_func=extract_X_Y_Lengths,
      batch_size=3,
      num_parallel_calls=3,
      # num_parallel_batches=FLAGS.num_threads_processing_data
  ))
  train_set = train_set.prefetch(buffer_size=3)
  iterator = train_set.make_initializable_iterator()
  a, b, c, d = iterator.get_next()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    while True:
      try:
        print(sess.run([tf.shape(a)]))
        print(sess.run(tf.log(0.5)/tf.log(10.0)))
      except tf.errors.OutOfRangeError:
        break
