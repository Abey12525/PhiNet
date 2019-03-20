import tensorflow as tf
import numpy as np
import InitRand as inr


rand_init = inr.RandomInit()
re = rand_init.rnn_reward_rnd_init()
layers,neuron_num = rand_init.neuro_rnd_init(np.random.randint(10,20))
neuron_num = np.array(neuron_num)
re = np.append(re,neuron_num)
shp = np.shape(re)
print(shp)
x = tf.placeholder(dtype=float,shape=[])
rnn = tf.contrib.cudann_rnn.CudnnLSTM(1,2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x = sess.run(rnn, feed_dict={x : re})
    print(x)
