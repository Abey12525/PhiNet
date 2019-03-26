import sys
print(sys.executable)


import tensorflow as tf
import numpy as np
import InitRand as inr


"""
__init__(
    num_layers,
    num_units,
    input_mode=CUDNN_INPUT_LINEAR_MODE,
    direction=CUDNN_RNN_UNIDIRECTION,
    dropout=0.0,
    seed=None,
    dtype=tf.dtypes.float32,
    kernel_initializer=None,
    bias_initializer=None,
    name=None
)
"""

tf.reset_default_graph()
#tf.contrib.training.bucket_by_sequence_lenght(x, dynamic_pad = true)
rand_init = inr.RandomInit()
re = rand_init.rnn_reward_rnd_init()
print("----",re,"----")
layers,neuron_num = rand_init.neuro_rnd_init(np.random.randint(10,20))
neuron_num.append(re)
neuron_num = np.array(neuron_num)
#tf.set_random_seed(10)
shp = neuron_num.shape[0]
print(shp)
x = tf.placeholder(dtype = tf.float32,shape=[None,1,shp])
# Culstm = tf.contrib.cudnn_rnn.CudnnLSTM(10,10)
# output, state = Culstm(x,initial_state = None, training = True)
# Culstm = tf.contrib.cudnn_rnn.CudnnLSTM(20,30)
#output1, state1 = Culstm(x,initial_state = None,training = True)
Culstm = tf.contrib.cudnn_rnn.CudnnLSTM(4,1)
output2,state2 = Culstm(x,initial_state = None,training = True)
Culstm2 = tf.contrib.cudnn_rnn.CudnnLSTM(4,15)
output3,state2 = Culstm2(output2,initial_state = state2, training = True)
init = tf.global_variables_initializer()
# with tf.variable_scope(reuse=tf.AUTO_REUSE):
#     x = sess.run(output, feed_dict={x : [re,re]})
#     print("Test Phase one complete ")
with tf.Session() as sess:

    sess.run(init)
    writer = tf.summary.FileWriter('./graphs1',sess.graph)
    xi = sess.run(output2,feed_dict={x : [[neuron_num]]})
    print(xi)
    print("##########################################")


