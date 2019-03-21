import sys
print(sys.executable)


import tensorflow as tf
import numpy as np
import InitRand as inr


#tf.contrib.training.bucket_by_sequence_lenght(x, dynamic_pad = true)

rand_init = inr.RandomInit()
re = rand_init.rnn_reward_rnd_init()
print("################################testing###################################")
layers,neuron_num = rand_init.neuro_rnd_init(np.random.randint(10,20))
neuron_num = np.array(neuron_num)
re = np.append(re,neuron_num)
#tf.set_random_seed(10)
tf.reset_default_graph()
x = tf.placeholder(dtype = tf.float32,shape=[None,shp[0]])
Culstm= tf.contrib.cudnn_rnn.CudnnLSTM(1,10)
output, state = Culstm(x,initial_state = None, training = True)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    x = sess.run(output, feed_dict={x : [re]})
    print("Test Phase one complete ")
    print(x)
