import tensorflow as tf
class TextCNN():
    def __init__(self,sequence_length,num_classes,vocab_size,embedding_size,filter_sizes,num_filters,l2_reg_lamdba):
        self.input_x = tf.placeholder(tf.int32,[None,sequence_length],name = 'input_x')
        self.input_y = tf.placeholder(tf.int32,[None,num_classes],name = 'input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32,name = 'dropout_keep_prob')

        l2_loss = tf.constant(0.0)

        with tf.name_scope('embedding'):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size,embedding_size],-1.0,1.0,name = 'W')
            )
            self.embedded_chars = tf.nn.embedding_lookup(self.W,self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars,-1)
        pooled_output = []
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s'%filter_size):
                filter_shape = [filter_size,embedding_size]