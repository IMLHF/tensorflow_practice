import tensorflow as tf
import numpy as np
import librosa
import os
import shutil

LEN_WAVE_PADDING_TO = 160000
SR=16000


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def load_wave_mat(speakerlist_dir):
  speaker_list = list(os.listdir(speakerlist_dir))
  speaker_mat = []
  for speaker_name in speaker_list:
    utt_mat = []
    speaker_dir = os.path.join(speakerlist_dir, speaker_name)
    if not os.path.isdir(speaker_dir):
      continue
    utt_list = list(os.listdir(speaker_dir))
    for utt_name in utt_list[:3]:
      utt_dir = os.path.join(speaker_dir, utt_name)
      waveData, sr = librosa.load(utt_dir,sr=SR)
      # waveData=(waveData / np.max(np.abs(waveData)))* 32767 # 数据在（-1,1）之间，在读取后变换为16bit，节省空间
      while len(waveData) < LEN_WAVE_PADDING_TO:
        waveData = np.tile(waveData, 2)
      utt_mat.append(waveData[:LEN_WAVE_PADDING_TO])
    speaker_mat.append(np.array(utt_mat, dtype=np.float32))
    # print(np.shape(np.array(utt_mat,dtype=np.float32)))
    print(speaker_name+' over')
  return np.array(speaker_mat, dtype=np.float32)


if __name__ == '__main__':
  speakerlist_dir = '/mnt/d/tf_recipe/ALL_DATA/aishell/mixed_data_small'
  speech = load_wave_mat(speakerlist_dir)
  speech_shape = np.shape(speech)
  speech=np.reshape(speech,[-1])
  # for i in range(speech_shape[0]):
  # wave_features = [tf.train.Feature(float_list=tf.train.FloatList(value=input_))
  #                  for input_ in speech[i]]

  print(np.shape(speech))
  # 1. writer
  writer = tf.python_io.TFRecordWriter(speakerlist_dir+'.tfrecords')

  # 2. example
  record = tf.train.Example(
      features=tf.train.Features(
          feature={
              # 'mat': _bytes_feature(speech.tostring())
              'shape': _int64_feature(speech_shape),
              'mat': _float_feature(speech)
          }))
  # serialize and write
  writer.write(record.SerializeToString())
  writer.close()
