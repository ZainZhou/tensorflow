import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
# from text_cnn import TextCNN
from tensorflow.contrib import learn

# Params
# Data loading params
tf.flags.DEFINE_float('dev_sample_percentage',1,'percentage of the training data to use for validation')
tf.flags.DEFINE_string('positive_data_file','./data/rt-polaritydata/rt-polarity.pos','Data source for the positive')
tf.flags.DEFINE_string('negative_data_file','./data/rt-polaritydata/rt-polarity.neg','Data source for the negative')

# Model Hparams
tf.flags.DEFINE_integer('embedding',128,'Dimensionality of character embedding')
tf.flags.DEFINE_string('filter_size','3,4,5','filter size')
tf.flags.DEFINE_integer('num_filters',128,'Number of filters')
tf.flags.DEFINE_float('dropout',0.5,'dropout')
tf.flags.DEFINE_float('L2_reg_lambda',0.0,'L2')

# Training params
tf.flags.DEFINE_integer('batch_size',128,'Batch size')
tf.flags.DEFINE_integer('num_epochs',200,'Number of epochs')
tf.flags.DEFINE_integer('evaluate_every',100,'evaluate_every')
tf.flags.DEFINE_integer('checkpoint_every',100,'Saving')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('/nParameters:')
for attr,value in sorted(FLAGS._flags.items()):
    print('%a:%v'%(attr,value))
