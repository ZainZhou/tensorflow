import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Params
# Data loading params
tf.flags.DEFINE_float('dev_sample_percentage',1,'percentage of the training data to use for validation')
tf.flags.DEFINE_string('positive_data_file','./data/rt-polaritydata/rt-polarity.pos','Data source for the positive')
tf.flags.DEFINE_string('negative_data_file','./data/rt-polaritydata/rt-polarity.neg','Data source for the negative')
tf.flags.DEFINE_boolean('allow_soft_placement',True,'allow_soft_placement')
tf.flags.DEFINE_boolean('log_device_placement',True,'log_soft_placement')
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

FLAGS = tf.app.flags.FLAGS
FLAGS._parse_flags()
for attr,value in sorted(FLAGS.__flags.items()):
    print(('{} = {}').format(attr.upper(),value))

# Load data
x_text,y = data_helpers.load_data_and_labels(FLAGS.positive_data_file,FLAGS.negative_data_file)

max_document_length = max(len(x.split(' ')) for x in x_text)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(vocab_processor.fit_transform(x_text))

np.random.seed(10)
shuffle_index = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_index]
y_shuffled = y[shuffle_index]

dev_sample_index = -1*int(FLAGS.dev_sample_percentage*float(len(y)))

x_train,x_dev = x_shuffled[:dev_sample_index],x_shuffled[dev_sample_index:]
y_train,y_dev = y_shuffled[:dev_sample_index],y_shuffled[dev_sample_index:]
print(('max_document_length:{:d}').format(len(vocab_processor.vocabulary_)))
print(('train/dev split:{:d}/{:d}').format(len(y_train),len(y_dev)))

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement = FLAGS.allow_soft_placement,
        log_device_placement = FLAGS.log_device_placement
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length = x_train.shape[1],
            num_classes = y_train.shape[1],
            vocab_size = len(vocab_processor.vocabulary_),
            embedding_size = FLAGS.embedding,
            filter_sizes = list(map(int,FLAGS.filter_size.split(','))),
            num_filters = FLAGS.num_filters,
            l2_reg_lamdba = FLAGS.L2_reg_lambda
        )