import tensorflow as tf

v1 = tf.Variable(tf.random_normal([1,2]),name = 'v1')
v2 = tf.Variable(tf.random_normal([2,3]),name = 'v2')

#init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    #sess.run(init_op)
    saver.restore(sess,'./model/model.ckpt')
    print("V1:",sess.run(v1))
    print("V2:",sess.run(v2))
    #saver_path = saver.save(sess,"./model/model.ckpt")