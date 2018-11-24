import tensorflow as tf
import numpy as np
import os
import shutil


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


datamat_dir = '_datamat'
datasetname_list = ['train', 'validation', 'test_cc']
tfrecord_dir = '_datatfrecord'
if os.path.exists(tfrecord_dir):
  shutil.rmtree(tfrecord_dir)

for dataset_name in datasetname_list:
  npy_set_dir = os.path.join(datamat_dir, dataset_name)
  tfrecord_set_dir = os.path.join(tfrecord_dir, dataset_name)
  os.makedirs(tfrecord_set_dir)
  npy_list = os.listdir(npy_set_dir)
  for npy_filename in npy_list:
    npy_name = npy_filename[:npy_filename.rfind('.')]
    npydata = np.load(os.path.join(npy_set_dir, npy_filename))

    # 1. writer
    writer = tf.python_io.TFRecordWriter(
        os.path.join(tfrecord_set_dir, npy_name+'.tfrecords'))

    # 2. example
    record = tf.train.Example(
        features=tf.train.Features(
            feature={
                # bytes
                'mat': _bytes_feature(npydata.tostring())
            }))
    # serialize and write
    writer.write(record.SerializeToString())
    writer.close()
