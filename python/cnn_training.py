""" 
 ---= 
 ---  Convolutional Neural Network Estimator for Action Recognition, built with tf.layers.
 ---  2017, federico.corradi@inilabs.com 
 ---  data preparation and tensor flow classifications
 ---
 ---= 
 """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import scipy as sp
tf.merge_all_summaries = tf.summary.merge_all
tf.train.SummaryWriter = tf.summary.FileWriter
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

from math import sqrt


root_folder = "/home/federico/NAS/HumanRecording/"

def put_kernels_on_grid (kernel, pad = 1):
  '''Visualize conv. filters as an image (mostly for the 1st layer).
  Arranges filters into a grid, with some paddings between adjacent filters.
  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
  Return:
    Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    
  Tensorflow versions:
    This code should work with versions 0.X
    For versions 1.0+ fix it as:
      1) rename tf.pack to tf.stack
      2) rename tf.image_summary to tf.summary.image
  '''
  # get shape of the grid. NumKernels == grid_Y * grid_X
  def factorization(n):
      for i in range(int(sqrt(float(n))), 0, -1):
          if n % i == 0:
              if i == 1: print('Who would enter a prime number of filters')
              return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
  print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)

  kernel1 = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel1.get_shape()[0] + 2 * pad
  X = kernel1.get_shape()[1] + 2 * pad

  channels = kernel1.get_shape()[2]

  # put NumKernels to the 1st dimension
  x2 = tf.transpose(x1, (3, 0, 1, 2))
  # organize grid on Y axis
  x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))

  # switch X and Y axes
  x4 = tf.transpose(x3, (0, 2, 1, 3))
  # organize grid on X axis
  x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x6 = tf.transpose(x5, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x7 = tf.transpose(x6, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return x7


def release_list(a):
   del a[:]
   del a


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features, [-1, 64, 64, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 64, 64, 1]
  # Output Tensor Shape: [batch_size, 64, 64, 32]
 
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=8,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      name="conv1")
      #kernel_initializer = True,
      #reuse = True)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 64, 64, 32]
  # Output Tensor Shape: [batch_size, 32, 32, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 32, 32, 64]
  # Output Tensor Shape: [batch_size, 32, 32, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=8,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      name="conv2")
  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 32, 32, 64]
  # Output Tensor Shape: [batch_size, 16, 16, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 16, 16, 64]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=16,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu,
      name="conv3")

  # Pooling Layer #3
  # Second max pooling layer with a 3x3 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 16, 16, 64]
  # Output Tensor Shape: [batch_size, 16 * 16 * 64]
  pool3_flat = tf.reshape(pool3, [-1, 7*7*16])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 16 * 16 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)



  
  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=33)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=33)
    print(onehot_labels.get_shape())
    #if(onehot_labels.get_shape() != (?,33)):
     #       print("label problem")
      #      print(onehot_labels)
       #     raise Exception
    # print(onehot_labels);
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

  # Build an initialization operation to run below.
  init = tf.initialize_all_variables()
  for t in tf.global_variables():
	print(t.name)

  def getKernel1():
        with tf.variable_scope("conv1", reuse=True) as scope_conv:
              W_conv1 = tf.get_variable('kernel', shape=[5, 5, 1, 8])
              #weights = W_conv1.eval()
              grid = put_kernels_on_grid (W_conv1)
              #ten, sum_op = tf.summary.image('conv1/kernels', grid, max_outputs=1)
            
              return tf.summary.image('conv1_kernels', grid, max_outputs=1)

  def getKernel2():
        with tf.variable_scope("conv2", reuse=True) as scope_conv:
              W_conv2 = tf.get_variable('kernel', shape=[5, 5, 8, 8])
              #weights = W_conv1.eval()
              grid = put_kernels_on_grid (W_conv2)
              #ten, sum_op = tf.summary.image('conv1/kernels', grid, max_outputs=1)
            
              return tf.summary.image('conv2_kernels', grid, max_outputs=1)

  def getKernel3():
        with tf.variable_scope("conv3", reuse=True) as scope_conv:
              W_conv3 = tf.get_variable('kernel', shape=[3, 3, 8, 16])
              #weights = W_conv1.eval()
              grid = put_kernels_on_grid (W_conv3)
              #ten, sum_op = tf.summary.image('conv1/kernels', grid, max_outputs=1)
            
              return tf.summary.image('conv3_kernels', grid, max_outputs=1)

  #Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }
  eval_metric_ops = {
      "accuracy_metric": tf.metrics.accuracy(
            labels,
            tf.argmax(
               input=logits, axis=1)
                      ), 
      "conv1_kernels": getKernel1()
      #"conv2_kernels": getKernel2()
      #"conv3_kernels": getKernel3()
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)
  for t in tf.global_variables():
	print(t.name)

      
def read_npy_chunk(filename, start_row, num_rows):
    """
    Reads a partial array (contiguous chunk along the first
    axis) from an NPY file.
    Parameters
    ----------
    filename : str
        Name/path of the file from which to read.
    start_row : int
        The first row of the chunk you wish to read. Must be
        less than the number of rows (elements along the first
        axis) in the file.
    num_rows : int
        The number of rows you wish to read. The total of
        `start_row + num_rows` must be less than the number of
        rows (elements along the first axis) in the file.
    Returns
    -------
    out : ndarray
        Array with `out.shape[0] == num_rows`, equivalent to
        `arr[start_row:start_row + num_rows]` if `arr` were
        the entire array (note that the entire array is never
        loaded into memory by this function).
    """
    assert start_row >= 0 and num_rows > 0
    with open(filename, 'rb') as fhandle:
        major, minor = numpy.lib.format.read_magic(fhandle)
        shape, fortran, dtype = numpy.lib.format.read_array_header_1_0(fhandle)
        assert not fortran, "Fortran order arrays not supported"
        # Make sure the offsets aren't invalid.
        assert start_row < shape[0], (
            'start_row is beyond end of file'
        )
        assert start_row + num_rows <= shape[0], (
            'start_row + num_rows > shape[0]'
        )
        # Get the number of elements in one 'row' by taking
        # a product over all other dimensions.
        row_size = numpy.prod(shape[1:])
        start_byte = start_row * row_size * dtype.itemsize
        fhandle.seek(start_byte, 1)
        n_items = row_size * num_rows
        flat = numpy.fromfile(fhandle, count=n_items, dtype=dtype)
        return flat.reshape((-1,) + shape[1:])



def main(unused_argv):
  #list_person = "/home/inilabs/NAS/HumanRecording/DATApython/list_persons.npy"
  #list_persons=np.load(list_person)
  scale_input_image_size_x = 64
  scale_input_image_size_y = 64
  camera_size_y = 261
  camera_size_x = 346
  for person in range(1,3):#len(list_persons)):
   #person_name = (list_persons[len(list_persons)-person-1]).split("/")[::-1][0]
   for counter in range(3,5):
    train_data_final = []
    train_labels_final = []

    train_filename_data = root_folder + "DATApython/DATAformatedForTraining2/cam_3_test_data_2.npy"

    train_filename_labels = root_folder + "DATApython/DATAformatedForTraining2/cam_3_test_labels_2.npy"

    test_dat = np.load(root_folder + "DATApython/DATAformatedForTraining2/cam_3_test_data_3.npy")

    test_lab = np.load(root_folder + "DATApython/DATAformatedForTraining2/cam_3_test_labels_3.npy")

    print(train_filename_data)
    try:
       train_data = np.load(train_filename_data)
       train_labels = np.load(train_filename_labels)
       print(train_data[0:int(0.7*len(train_data))])
       print(train_labels)
        #train_data = read_npy_chunk(train_filename_data,int(0+counter*0.01*len(train_data)),int((counter+1)*0.01*len(train_data))
        #train_labels = read_npy_chunk(train_filename_labels, int(0+counter*0.01*len(train_labels)),int((counter+1)*0.01*len(train_labels))
    except:
        print("file nor found")
        #print(train_data.size())
        #print(train_labels.size())
     #for index in range(int(0+counter*0.01*len(train_data)),int((counter+1)*0.01*len(train_data))):
    train_data_final = train_data[0:int(0.7*len(train_data))]
    train_labels_final = train_labels[0:int(0.7*len(train_labels))]
          #print(train_data_final.shape)
          #print(train_labels_final.shape)
      

    un_labels = np.unique(train_labels_final)
    if(len(un_labels)!=33):
      print("label problem-we resize the label vector to 33")
      #print(train_labels)
      #print(un_labels)
      un_labels = [11,12,13,14,15,16,17,18,21,22,23,24,25,26,31,32,33,34,35,36,41,42,43,44,45,46,51,52,53,54,55,56,57]
      #print(un_labels)
      #raise Exception
   
    for this_label in range(len(un_labels)):
      tf_index = train_labels_final == un_labels[this_label]
      print(train_labels_final)
      train_labels_final[tf_index] = this_label

    # Scale Input from png 0,255 to 0,1
    train_data_f = []
    for i in range(len(train_data_final)):
      si = (np.reshape(train_data_final[i], [camera_size_y,camera_size_x])).copy().astype(np.float32)
      ss = sp.misc.imresize(si,(scale_input_image_size_x,scale_input_image_size_y)).astype(np.float32)
      train_data_f.append(ss.reshape([scale_input_image_size_x*scale_input_image_size_y])/255.0)
    train_data_f = np.array(train_data_f, dtype="float32")
    train_data_final = train_data_f

    # Scale Input from png 0,255 to 0,1
    #eval_dat = np.load(test_filename_data)
    eval_data = test_dat
    
    #eval_label = np.load(test_filename_labels)
    eval_labels = test_lab
    
    for this_label in range(len(un_labels)):
      tf_index = eval_labels == un_labels[this_label]
      eval_labels[tf_index] = this_label
    eval_data_f = []
    for i in range(len(eval_data)):
      si = (np.reshape(eval_data[i], [camera_size_y,camera_size_x])).copy().astype(np.float32)
      ss = sp.misc.imresize(si,(scale_input_image_size_x,scale_input_image_size_y)).astype(np.float32)
      eval_data_f.append(ss.reshape([scale_input_image_size_x*scale_input_image_size_y])/255.0)
    eval_data_f = np.array(eval_data_f, dtype="float32")
    eval_data = eval_data_f


  
  # Create the Estimator
    human_skeleton_classifier = learn.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/human_skeleton_model_33V12")
  # Initialize variables
    #conv1= tf.get_variable("conv1")
    #conv2= tf.get_variable("conv2")
    #conv3= tf.get_variable("conv3")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    ##with tf.variable_scope("conv1"):
              ##tf.get_variable_scope().reuse_variables()
              ##weights = tf.get_variable('weights')
    #with tf.variable_scope('conv1', reuse=True) as scope_conv:
              #W_conv1 = tf.get_variable('weights', shape=[5, 5, 1, 32])
              ##weights = W_conv1.eval()
              #grid = put_kernels_on_grid (W_conv1)
              #summary_saver = tf.contrib.learn.monitors.SummarySaver(
                  #tf.summary.image('conv1_kernels', grid, max_outputs=1),
                  #save_steps=50,
                  #output_dir="/tmp/human_skeleton_model_33V7")

    validation_metrics = {
    "accuracy":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_accuracy,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
    "precision":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_precision,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
    "recall":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_recall,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES)#,
    #"conv1/kernels":
       #tf.contrib.learn.MetricSpec(
            #metric_fn=getKernel("conv1"),
            #prediction_key=tf.contrib.learn.PredictionKey.CLASSES)
}

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        eval_data,
        eval_labels,
        every_n_steps=1,
        metrics=validation_metrics)


   # Train the model
    human_skeleton_classifier.fit(
				  x=train_data_final,
				  y=train_labels_final,
				  batch_size=32,
				  steps=400000,
				  monitors=[logging_hook,validation_monitor])
 

 
  #print("evall"+eval_results)
  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }

  # Evaluate the model and print results
  eval_results = human_skeleton_classifier.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)

  # Visualize conv1 features
  with tf.variable_scope('conv1', reuse=True) as scope_conv:
     W_conv1 = tf.get_variable('weights', shape=[5, 5, 1, 8])
     weights = W_conv1.eval()
     with open("conv1.weights.npz", "w") as outfile:
        np.save(outfile, weights)

if __name__ == "__main__":
  tf.app.run()

