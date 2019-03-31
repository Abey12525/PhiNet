import tensorflow as tf
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
tf.reset_default_graph
saver = tf.train.Saver()
x = tf.placeholder(dtype=tf.float32,shape=[None, 1, 1])
c = tf.contrib.cudnn_rnn.CudnnLSTM(10,5)
output, state = c(x,initial_state = None, training = True)
d = tf.contrib.cudnn_rnn.CudnnLSTM(10,4)
out,st = d(output,initial_state = None, training = True)
e = tf.contrib.cudnn_rnn.CudnnLSTM(6,3)
out1,st1 = e(out,initial_state = None, training = True)
f = tf.contrib.cudnn_rnn.CudnnLSTM(10,1)
out2,st2 = f(out1,initial_state = None,training = True)

"""
tf.contrib.layers.fully_connected(
    inputs,
    num_outputs,
    activation_fn=tf.nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None
)
"""
with tf.Session() as sess:
    Writer = tf.summary.FileWriter(logdir='./graphs/G{}/'.format(timestr),graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    x = sess.run(out2,feed_dict={x : [[[20]]]})
    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)
    print(x)
