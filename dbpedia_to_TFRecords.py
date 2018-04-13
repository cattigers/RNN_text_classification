from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import pandas
import tensorflow as tf
import pdb

def np_to_tfrecords(X, Y, file_path_prefix, verbose=True):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value.reshape(-1)))

    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print ("Serializing {:d} examples into {}".format(X.shape[0], result_tf_file))
        
    # iterate over each sample,
    # and serialize it as ProtoBuf.
    for idx in range(X.shape[0]):
        #pdb.set_trace()
        example = tf.train.Example(features=tf.train.Features(feature={
        'X': _int64_feature(X[idx]),
        'Y': _int64_feature(Y[idx])}))
        
        serialized = example.SerializeToString()
        writer.write(serialized)
    
    if verbose:
        print ("Writing {} done!".format(result_tf_file))
        

MAX_DOCUMENT_LENGTH=200

def main():
    # Prepare training and testing data
    # Prepare training and testing data
    print('MAX_DOCUMENT_LENGTH', MAX_DOCUMENT_LENGTH)
    dbpedia = tf.contrib.learn.datasets.load_dataset(
      'dbpedia', size='large', test_with_fake_data=False)
    
    print("Shuffling data set...")
    x_train = dbpedia.train.data[:, 1]
    y_train = dbpedia.train.target
    s = np.arange(len(y_train))
    np.random.shuffle(s)
    x_train = x_train[s]
    y_train = y_train[s]
    print("Done!")  
    
    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(dbpedia.test.data[:, 1])
    y_test = pandas.Series(dbpedia.test.target)
    
    print('Train data size:', x_train.shape)
    print('Test data size:', x_test.shape)
    
    # Process vocabulary
    char_processor = tf.contrib.learn.preprocessing.ByteProcessor(
      MAX_DOCUMENT_LENGTH)
    x_train_fit = np.array(list(char_processor.fit_transform(x_train)), dtype=np.int)
    x_test_fit = np.array(list(char_processor.transform(x_test)), dtype=np.int)
    
    y_train = np.expand_dims(np.asarray(y_train), axis=1) 
    y_test = np.expand_dims(np.asarray(y_test), axis=1) 
  
    np_to_tfrecords(x_train_fit, np.asarray(y_train, np.int), 'train', verbose=True)
    np_to_tfrecords(x_test_fit, np.asarray(y_test, np.int), 'test', verbose=True)
    
    total_err = 0
    err = 0
    for i, serialized_example in enumerate(tf.python_io.tf_record_iterator('train.tfrecords')):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        x_1 = np.array(example.features.feature['X'].int64_list.value)
        y_1 = np.array(example.features.feature['Y'].int64_list.value)
        err += np.linalg.norm(x_train_fit[i]-x_1) + np.linalg.norm(y_train[i]-y_1)
        total_err += err
        if err>0:
            pass
            #break
    print('Train set Error: %f'% total_err)
    

    
    err = 0
    for i, serialized_example in enumerate(tf.python_io.tf_record_iterator('test.tfrecords')):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        x_1 = np.array(example.features.feature['X'].int64_list.value)
        y_1 = np.array(example.features.feature['Y'].int64_list.value)
        err += np.linalg.norm(x_test_fit[i]-x_1) + np.linalg.norm(y_test[i]-y_1)    
    print('Test set Error: %f'% err)
    
if __name__ == '__main__':
    main()    
