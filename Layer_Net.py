import tensorflow as tf
import numpy as np
import InitRand as inr
import time


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

class Layer():
    def __init__(self,inp_x,shape):
        self.inp = inp_x #-->  reward + action
        #self.state = inp_x[:-1]
        self.shape = shape

    def Model(self,discount_factor=0.8,learning_rate=0.001):
        x = tf.placeholder(dtype=tf.float32,shape =[None,1,self.shape])
        linear_layer0 = tf.get_variable("wl0",shape=[1,5],initializer=tf.contrib.layers.xavier_initializer())
        linear_layer = tf.get_variable("wl1",shape=[5,1],initializer=tf.contrib.layers.xavier_initializer())
        linear_bias0 = tf.get_variable("bl0",shape = [1],initializer = tf.random_uniform_initializer(2,10))
        linear_bias = tf.get_variable("bl",shape = [1],initializer = tf.random_uniform_initializer(2,10))

        Culstm = tf.contrib.cudnn_rnn.CudnnLSTM(6,self.shape)
        output1, state1 = Culstm(x,initial_state = None,training = True)
        Culstm1 = tf.contrib.cudnn_rnn.CudnnLSTM(6,self.shape+4)
        output2,state2 = Culstm1(output1,initial_state = state1,training = True)
        Culstm2 = tf.contrib.cudnn_rnn.CudnnLSTM(6,self.shape+4)
        output3,state3 = Culstm2(output2,initial_state = state2, training = True)
        Culstm3 = tf.contrib.cudnn_rnn.CudnnLSTM(6,self.shape-2)
        output4,state4 = Culstm3(output3,initial_state = state3, training = True)
        linear = tf.add(tf.matmul(tf.transpose(output4[-1]),linear_layer0),linear_bias0)
        output = tf.argmax(tf.add(tf.matmul(linear,linear_layer),linear_bias))
        #output shape = 1 !! I think

        #optimizer = tf.train.GradientDescentOptimizer(output).minimize(-self.reward)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            Writer = tf.summary.FileWriter(logdir='./graphs/G{}/'.format(timestr), graph=sess.graph)
            sess.run(init)
            output = sess.run(output,feed_dict={x : self.inp})
            print(output)
        return output

    def run(self):
        """
            'inr' is used to initialize a random reward , random number of layers and
            random number of neurons in each layers
            """
        try:
            """Checking if file exist or not"""
            neuron_num = np.load('./layers.npy')
            re = np.load('./reward.npy')
            neuron_num = neuron_num[0]
            re = re[0]
            neuron_num.append(re)
            shape = neuron_num.shape[0]
        except:
            print("Structure Not Found !!")
            print("NOTE: initialized or file not written in the correct format")
        else:
            # initializing RandomInit class
            rand_init = inr.RandomInit()
            # initializing Random Reward --> eg : 10
            re = rand_init.rnn_reward_rnd_init()
            # initializing Random number of layers and neuron in each layers --> eg : [250,200,100]
            layers, neuron_num = rand_init.neuro_rnd_init(np.random.randint(10, 20))
            # appending reward to the list of number of neurons in each layers --> eg: [250,200,100,10]
            neuron_num.append(re)
            # converting to numpy array
            neuron_num = np.array(neuron_num)
            # tf.set_random_seed(10)
            shape = neuron_num.shape[0]
        """
        initializing the lstm model
        passing neuron_num as 3D array since CudnnLSTM is time major  
        """
        # lstm_output   - Output of the Model
        # lstm_state    - State of the Model
        lstm_output = np.round(self.Model())
        np.save('./layers', lstm_output)
        print(lstm_output)


# if __name__=='__main__':
#     """
#     'inr' is used to initialize a random reward , random number of layers and
#     random number of neurons in each layers
#     """
#     try:
#         """Checking if file exist or not"""
#         neuron_num = np.load('./layers.npy')
#         re = np.load('./reward.npy')
#         neuron_num = neuron_num[0]
#         re = re[0]
#         neuron_num.append(re)
#         shape = neuron_num.shape[0]
#     except:
#         print("Structure Not Found !!")
#         print("NOTE: initialized or file not written in the correct format")
#     else:
#         #initializing RandomInit class
#         rand_init = inr.RandomInit()
#         #initializing Random Reward --> eg : 10
#         re = rand_init.rnn_reward_rnd_init()
#         #initializing Random number of layers and neuron in each layers --> eg : [250,200,100]
#         layers, neuron_num = rand_init.neuro_rnd_init(np.random.randint(10, 20))
#         #appending reward to the list of number of neurons in each layers --> eg: [250,200,100,10]
#         neuron_num.append(re)
#         #converting to numpy array
#         neuron_num = np.array(neuron_num)
#         #tf.set_random_seed(10)
#         shape = neuron_num.shape[0]
#     """
#     initializing the lstm model
#     passing neuron_num as 3D array since CudnnLSTM is time major
#     """
#     lstm = Layer([[neuron_num]],shape)
#     # lstm_output   - Output of the Model
#     # lstm_state    - State of the Model
#     lstm_output=np.round(lstm.Model())
#     np.save('./layers',lstm_output)
#     print(lstm_output)
