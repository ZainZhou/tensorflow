import tensorflow as tf
class TextCNN():
    def __init__(self,sequence_length,num_classes,vocab_size,embedding_size,filter_sizes,num_filters,l2_reg_lamdba):
        self.input_x = tf.placeholder(tf.int32,[None,sequence_length],name = 'input_x')
