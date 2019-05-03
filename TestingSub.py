import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.datasets import mnist

"""
    1. initial condition of weights - Done 
        i) masking condition ? - Done
    2. weights after first initialization - Done
    3. initial condition of layer input x - Done
    4. layer after initial condition - N/A 
    5. mask initialization - Done
    6. mask after initial state -Done
    7.output layer - N/A
"""
class PChild():
    def __init__(self,train_x,train_y=None):
        """Reading the child network structure"""
        try :
            self.neurons = np.load('./neurons.npy')
            self.neurons = self.neurons.astype(int)
            self.neurons = self.neurons[0]
            self.neuron_mask = np.load('./mask.npy')
            self.no_layers = len(self.neurons)
            self.neuron_num = sum(self.neurons)
            self.neurons = self.neurons.astype(int)
            self.neuron_mask = self.neuron_mask.astype(int)
            self.data_len = len(train_x)
            print("data_length: ",self.data_len)
            sys.stdout.write("Structure file found!!!\n")
            sys.stdout.flush()
        except:
            print("Structure Files not Found")

        """creating placeholder for the shape of the input and output"""
        tf.reset_default_graph()
        self.train_x = train_x/255
        self.shape_x = []
        self.tmp_shp_x = tf.shape(train_x)
        if train_y.any():
            self.shape_y = []
            self.train_y = train_y
            self.tmp_shp_y = tf.shape(train_y)
            with tf.Session() as sess:
                out_y = sess.run(self.tmp_shp_y)
                out_y = out_y[1:]
                self.shape_y.append(None)
                for j in out_y:
                    self.shape_y.append(j)
                out = sess.run(self.tmp_shp_x)
                out = out[1:]
                self.shape_x.append(None)
                for x in out:
                    self.shape_x.append(x)
        else:
            print("Unsupervised learning is Currently not supported!!")
            exit()
            # with tf.Session() as sess:
            #     out = sess.run(self.tmp_shp_x)
            #     self.shape_x.append(None)
            #     for x in out:
            #         self.shape_x.append(x)
            #     self.x_inp = tf.placeholder(tf.float32, self.shape_x)
            #     print("x : ", self.x_inp)

        sys.stdout.write("placeholder initialized\n")
        sys.stdout.flush()
        """Initializing the child network"""
        weights = []
        bias = []
        mask = []
        # Input layer and Input weights
        tf.reset_default_graph()
        self.Linp = tf.get_variable("Linp",shape = out,initializer = tf.ones_initializer)
        self.Winp = tf.get_variable("Winp",shape = [out,self.neurons[0]],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Binp = tf.get_variable("Binp",shape = [1], initializer=tf.contrib.layers.xavier_initializer())
        count = 0
        for i, Neurons in enumerate(self.neurons):
            vars()["w{}".format(i)] = tf.get_variable("W{}".format(i), shape=[Neurons,self.neuron_num],
                                                      initializer=tf.contrib.layers.xavier_initializer())
            vars()["b{}".format(i)] = tf.get_variable("b{}".format(i), shape=[1],
                                                      initializer = tf.contrib.layers.xavier_initializer())
            vars()["wm{}".format(i)] = self.neuron_mask[count:count+Neurons]
            count += Neurons
            W_str = "w{}".format(i)
            W_tmp = eval(W_str)
            B_str = "b{}".format(i)
            B_tmp = eval(B_str)
            WM_str = "wm{}".format(i)
            WM_tmp = eval(WM_str)
            weights.append((i,W_tmp))
            bias.append((i,B_tmp))
            mask.append(WM_tmp)

        self.Wout = tf.get_variable("Wout", shape = [self.neurons[-1],out_y],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Bout = tf.get_variable("Bout", shape = [1],initializer=tf.contrib.layers.xavier_initializer())
        self.init = tf.global_variables_initializer()
        self.weights = dict(weights)
        self.bias = dict(bias)
        self.mask = mask
        sys.stdout.write("structure initialization complete\n")
        sys.stdout.flush()



    def model(self,iteration,batch_size=500,learning_rate=0.9):
        self.iteration = iteration
        self.batch_size = batch_size
        self.total_batch = self.data_len // self.batch_size
        #weights = dict(self.weights)
        """"placeholders"""
        lyr_no = tf.placeholder(tf.int32,[None,1])
        lyr_p = tf.placeholder(tf.float32,[None,self.neuron_num])
        self.x_inp = tf.placeholder(tf.float32, self.shape_x)
        self.y_inp = tf.placeholder(tf.float32, self.shape_y)

        print("total_batch : ",self.total_batch)
        """numpy , input batch"""
        self.processed_inp = []
        lyr_str = []
        layers = []

        inp_pro = self.x_inp
        #inp_pro = tf.nn.tanh(tf.add(inp_pro,self.Binp))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            count = 0
            for i in range(self.total_batch):
                data = self.train_x[count:count+self.batch_size]
                count += self.batch_size
                for j in range(self.batch_size):
                    inp_pr = sess.run(inp_pro, feed_dict={self.x_inp : [data[j]]})
                    print(inp_pr)
                    self.processed_inp.append(inp_pr)
                    if j == 10:
                        break
                if i == 1:
                    break

                # for i,p_inp in enumerate(self.processed_inp):
                #     layer_cpy = self.layers
                #     layer_cpy[0] = p_inp
                #     inp_str.append(layer_cpy)
                # for lyr in range(self.no_layers):
                #     #layer = np.concatenate(layer_cpy, 0)
                #     # actual_weights = sess.run(tf.multiply(self.weights[lyr], self.mask[lyr]))
                #     # actual_layer = sess.run(tf.multiply(self.mask[lyr], layer))  # correct
                #     # actual_layer_inp = sess.run(tf.multiply(actual_weights, actual_layer))  # check multiplication
                #     # actual_layer_inp = sess.run(tf.reduce_sum(actual_layer_inp, 1))
                #     # layer_result = sess.run(tf.add(actual_layer_inp, self.bias[lyr]))  # correct
                #     #layer_cpy[lyr] = sess.run(tf.nn.tanh(layer_result))
                #     layer_cpy[lyr] = sess.run(tf.add(layer_cpy[lyr],1))
                #     #print(layer_cpy[lyr])
                #     t.append(layer_cpy[lyr])
                #     print(lyr)
        lyr_cpy = []
        for i,Neurons in enumerate(self.neurons):
            if i == 0:
                pass
            else:
                L_tmp = np.ones(Neurons).tolist()
                layers.append(L_tmp)
        self.layers = layers
        for im,p_cnt in enumerate(self.processed_inp):
            layer = []
            lyr_cpy = self.layers
            layer.append(p_cnt)
            for lyr in range(self.no_layers):
                if lyr == 0:
                    layer.append(lyr_cpy)
                print(p_cnt)
            if im == 5:
                break


    def test_accuracy(self):
        #test accuray using the weights saved
        np.save('./accuracy.npy',self.accuracy)




"""
        y_i = x * D_i * W_i = [....x....] * diag([....d....]) * transpose([....w....])
        neurons must be initialized in order to create the network 
        //work in progress 
        for neurons in self.neurons:
            first layer = Neurons
            second layer = Neurons in first layer + Neurons
            third layer = Neurons in second layer + Neurons 
            fourth layer = Neurons in third layer + Neurons 
"""


if __name__ == '__main__':
    (train_x,train_y),(test_x,test_y) = mnist.load_data()
    train_x = np.reshape(train_x,[-1,784])
    y = tf.one_hot(test_y,depth = 10)
    with tf.Session() as sess:
        train_y = sess.run(y)
    child = PChild(train_x,train_y)
    child.model(iteration = 1)
    #print("Accuracy recorded")