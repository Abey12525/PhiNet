import tensorflow as tf
import numpy as np

class Neurons():
    def __init__(self,Layers):
        self.Layers = Layers
        #random init of neurons per layer
        self.inp = np.random.uniform(low=200,high=1000,size = (1,1,self.Layers))

    def Model(self):
        x = tf.placeholder(dtype=tf.float32,shape=[None,1,self.Layers])
        clstm1 = tf.contrib.cudnn_rnn.CudnnLSTM(20,10)
        output1,state1 = clstm1(x,initial_state = None, training = True)
        clstm2 = tf.contrib.cudnn_rnn.CudnnLSTM(15,8)
        output2,state2 = clstm2(output1,initial_state = state1, training = True)
        clstm3 = tf.contrib.cudnn_rnn.CudnnLSTM(10,5)
        output3,state3 = clstm3(output2, initial_state = state2,training = True)
        clstm4 = tf.contrib.cudnn_rnn.CudnnLSTM(5,self.Layers)
        model_output,model_state = clstm4(output3, initial_state = None,training = True)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            output = sess.run(model_output,feed_dict={x : self.inp})
        return output


if __name__ == '__main__':
    try:
        foo = np.load('./layers.npy')
        print(foo)
    except:
        print("File not Found -- !!")
    Neu = Neurons(10)
    out = Neu.Model()
    np.save('./neurons',out)
    print(out)