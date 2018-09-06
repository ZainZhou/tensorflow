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
tf.flags.DEFINE_float('dev_sample_percentage',0.15,'percentage of the training data to use for validation')
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
tf.flags.DEFINE_integer('num_checkpoints',5,'Saving')

FLAGS = tf.app.flags.FLAGS
FLAGS._parse_flags()
for attr,value in sorted(FLAGS.__flags.items()):
    print(('{} = {}').format(attr.upper(),value))

# Load data
x_text,y = data_helpers.load_data_and_labels(FLAGS.positive_data_file,FLAGS.negative_data_file)
max_document_length = max(len(x.split(' ')) for x in x_text)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
np.random.seed(10)
shuffle_index = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_index]
y_shuffled = y[shuffle_index]
dev_sample_index = -1*int(FLAGS.dev_sample_percentage*float(len(y)))
print(dev_sample_index)
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
        global_step = tf.Variable(0,name = 'global_step')
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars,global_step)

        grad_summaries = []
        for g,v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name),g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name),tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir,'runs',timestamp))
        print("Writing to {}\n".format(out_dir))
        loss_summary = tf.summary.scalar('loss',cnn.loss)
        acc_summary = tf.summary.scalar('acc',cnn.accuracy)
        #train summaries
        train_summary_op = tf.summary.merge([loss_summary,acc_summary,grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir,'summary','train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir,sess.graph)
        #dev summaries
        dev_summary_op = tf.summary.merge([loss_summary,acc_summary])
        dev_summary_dir = os.path.join(out_dir,'summaries','dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir,sess.graph)
        #Checkpoint directory
        checkpoint_dir = os.path.abspath(os.path.join(out_dir,'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir,'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.global_variables(),max_to_keep=FLAGS.num_checkpoints)

        vocab_processor.save(os.path.join(out_dir,'vocab'))

        sess.run(tf.global_variables_initializer())
        batches = data_helpers.batch_iter(list(zip(x_train,y_train)),FLAGS.batch_size,FLAGS.num_epochs)
        def train_step(x_batch,y_batch):
            feed_dic = {
                cnn.input_x : x_batch,
                cnn.input_y : y_batch,
                cnn.dropout_keep_prob : FLAGS.dropout
            }
            _, step, loss, accuracy = sess.run(
                [train_op,global_step,cnn.loss,cnn.accuracy],
                feed_dic
            )
            time_str = datetime.datetime.now().isoformat()
            print(('{}:step {},loss{:g},acc{:g}').format(time_str,step,loss,accuracy))

        def dev_step(x_batch,y_batch):
            feed_dic = {
                cnn.input_x : x_batch,
                cnn.input_y : y_batch,
                cnn.dropout_keep_prob : 1.0
            }
            _, step, loss, accuracy = sess.run(
                [train_op,global_step,cnn.loss,cnn.accuracy],
                feed_dic
            )
            time_str = datetime.datetime.now().isoformat()
            print(('{}:step {},loss{:g},acc{:g}').format(time_str,step,loss,accuracy))

        for batch in batches:
            x_batch,y_batch = zip(*batch)
            train_step(x_batch,y_batch)
            current_step = tf.train.global_step(sess,global_step)
            if current_step % FLAGS.evaluate_every == 0 :
                print('\n evaluation_every')
                dev_step(x_dev,y_dev)
            if current_step % FLAGS.checkpoint_every == 0 :
                path = saver.save(sess,'./model',global_step = current_step)
                print('model saved')