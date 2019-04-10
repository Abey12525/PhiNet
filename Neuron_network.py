import tensorflow as tf
import numpy as np

class Neurons():
    def __init__(self,Layers):
        self.Layers = Layers
        #random init of neurons per layer
        self.inp = np.random.uniform(low=200,high=1000,size = (1,1,self.Layers))

    def Model(self):
        tf.reset_default_graph()
        x = tf.placeholder(dtype=tf.float32,shape=[None,1,self.Layers])
        linear_layer0 = tf.get_variable("wln0", shape=[self.Layers,1], initializer=tf.contrib.layers.xavier_initializer())
        linear_layer = tf.get_variable("wln1", shape=[10, 1], initializer=tf.contrib.layers.xavier_initializer())

        linear_bias0 = tf.get_variable("bln0", shape=[1], initializer=tf.random_uniform_initializer(1,2))
        linear_bias = tf.get_variable("bln", shape=[self.Layers], initializer=tf.random_uniform_initializer(500, 900))
        clstm1 = tf.contrib.cudnn_rnn.CudnnLSTM(20,self.Layers,bias_initializer = tf.random_uniform_initializer(10,20))
        output1,state1 = clstm1(x,initial_state = None, training = True)
        clstm2 = tf.contrib.cudnn_rnn.CudnnLSTM(15,self.Layers+4,bias_initializer = tf.random_uniform_initializer(10,20))
        output2,state2 = clstm2(output1,initial_state = state1, training = True,)
        clstm3 = tf.contrib.cudnn_rnn.CudnnLSTM(10,self.Layers+4,dropout = 0.5,bias_initializer = tf.random_uniform_initializer(10,20))
        output3,state3 = clstm3(output2, initial_state = state2,training = True)
        clstm4 = tf.contrib.cudnn_rnn.CudnnLSTM(5,self.Layers,bias_initializer = tf.random_uniform_initializer(10,20))
        model_output,model_state = clstm4(output3, initial_state = None,training = True)
        output = tf.round(tf.add(tf.matmul(model_output[0],linear_layer0),linear_bias))

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            output = sess.run(output,feed_dict={x : self.inp})
        return output


if __name__ == '__main__':
    try:
        foo = np.load('./layers.npy')
        print(foo)
    except:
        print("File not Found -- !!")
    Neu = Neurons(int(foo))
    out = Neu.Model()
    np.save('./neurons',out)
    print(out)