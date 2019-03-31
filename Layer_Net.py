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
#tf.contrib.training.bucket_by_sequence_lenght(x, dynamic_pad = true)

class Lstm():
    def __init__(self,inp_x,shape):
        self.inp_x = inp_x
        self.shape = shape

    def Model(self):
        x = tf.placeholder(dtype=tf.float32,shape =[None,1,self.shape])
        Culstm = tf.contrib.cudnn_rnn.CudnnLSTM(25,10)
        output1, state1 = Culstm(x,initial_state = None,training = True,activation = tf.nn.tanh)
        Culstm1 = tf.contrib.cudnn_rnn.CudnnLSTM(15,10)
        output2,state2 = Culstm1(output1,initial_state = state1,training = True)
        Culstm2 = tf.contrib.cudnn_rnn.CudnnLSTM(10,5)
        output3,state3 = Culstm2(output2,initial_state = state2, training = True)
        Culstm3 = tf.contrib.cudnn_rnn.CudnnLSTM(20,1)
        output4,state4 = Culstm3(output3,initial_state = None, training = True)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            output,state = sess.run([output4, state4],feed_dict={x : self.inp_x})
        return output,state


# with tf.variable_scope(reuse=tf.AUTO_REUSE):
#     x = sess.run(output, feed_dict={x : [re,re]})
#     print("Test Phase one complete ")

if __name__=='__main__':
    rand_init = inr.RandomInit()
    re = rand_init.rnn_reward_rnd_init()
    layers, neuron_num = rand_init.neuro_rnd_init(np.random.randint(10, 20))
    neuron_num.append(re)
    neuron_num = np.array(neuron_num)
    # tf.set_random_seed(10)
    shape = neuron_num.shape[0]
    lstm = Lstm([[neuron_num]],shape)
    lstm_output,lstm_state = lstm.Model()
    print(lstm_output)
