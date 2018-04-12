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
    """
    Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
    For supervised learning, feed training inputs to X and training labels to Y.
    For unsupervised learning, only feed training inputs to X, and feed None to Y.
    The length of the first dimensions of X and Y should be the number of samples.
    
    Parameters
    ----------
    X : numpy.ndarray of rank 2
        Numpy array for training inputs. Its dtype should be float32, float64, or int64.
        If X has a higher rank, it should be rshape before fed to this function.
    Y : numpy.ndarray of rank 2 or None
        Numpy array for training labels. Its dtype should be float32, float64, or int64.
        None if there is no label array.
    file_path_prefix : str
        The path and name of the resulting tfrecord file to be generated, without '.tfrecords'
    verbose : bool
        If true, progress is reported.
    
    Raises
    ------
    ValueError
        If input type is not float (64 or 32) or int.
    
    """
    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """        
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:  
            raise ValueError("The input should be numpy ndarray. \
                               Instaed got {}".format(ndarray.dtype))
            
    #pdb.set_trace()
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 2  # If X has a higher rank, 
                               # it should be rshape before fed to this function.
    assert isinstance(Y, np.ndarray) or Y is None
    
    # load appropriate tf.train.Feature class depending on dtype
    dtype_feature_x = _dtype_feature(X)
    if Y is not None:
        assert X.shape[0] == Y.shape[0]
        assert len(Y.shape) == 2
        dtype_feature_y = _dtype_feature(Y)            
    
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print ("Serializing {:d} examples into {}".format(X.shape[0], result_tf_file))
        
    # iterate over each sample,
    # and serialize it as ProtoBuf.
    for idx in range(X.shape[0]):
        x = X[idx]
        if Y is not None:
            y = Y[idx]
        
        d_feature = {}
        d_feature['X'] = dtype_feature_x(x)
        if Y is not None:
            d_feature['Y'] = dtype_feature_y(y)
            
        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
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