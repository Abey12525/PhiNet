import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

"""
    1. initial condition of weights - Done 
        i) masking condition ? - N/A
    2. weights after first initialization - N/A
    3. initial condition of layer input x - N/A
    4. layer after initial condition - N/A 
    5. mask initialization - Done
    6. mask after initial state -N/A
    7.output layer - N/A
"""
class PChild():
    def __init__(self):
        try :
            self.neurons = np.load('./neurons.npy')
            self.neurons = self.neurons.astype(int)
            self.neurons = self.neurons[0]
            self.neuron_mask = np.load('./mask.npy')
            self.layers = len(self.neurons)
            self.neuron_num = sum(self.neurons)
            self.neurons = self.neurons.astype(int)
            self.neuron_mask = self.neuron_mask.astype(int)
        except:
            print("Structure Files not Found")
        weights = []
        layers = []
        bias = []
        mask = []
        count = 0
        tf.reset_default_graph()
        for i, Neurons in enumerate(self.neurons):
            vars()["w{}".format(i)] = tf.get_variable("W{}".format(i), shape=[self.neuron_num,Neurons],
                                                      initializer=tf.contrib.layers.xavier_initializer())
            vars()["b{}".format(i)] = tf.get_variable("b{}".format(i), shape=[1],initializer = tf.contrib.layers.xavier_initializer())
            vars()["L{}".format(i)] = tf.get_variable("L{}".format(i),shape = [Neurons], initializer= tf.zeros_initializer())
            vars()["wm{}".format(i)] = self.neuron_mask[count:count+Neurons]
            count += Neurons
            W_str = "w{}".format(i)
            W_tmp = eval(W_str)
            L_str = "L{}".format(i)
            L_tmp = eval(L_str)
            B_str = "b{}".format(i)
            B_tmp = eval(B_str)
            WM_str = "wm{}".format(i)
            WM_tmp = eval(WM_str)
            weights.append(W_tmp)
            layers.append(L_tmp)
            bias.append(B_tmp)
            mask.append(WM_tmp)
        print("WEIGHTS")
        print(weights)
        print("LAYERS")
        print(layers)
        print("BIAS")
        print(bias)
        print("MASK")
        print(mask)
        self.weights = weights
        self.layers = layers
        self.bias = bias
        self.mask = mask

    def fit(self,train_x,train_y = None):
        self.train_x = train_x
        self.shape_x = []
        self.tmp_shp_x = tf.shape(train_x)
        if train_y.any():
            self.shape_y = []
            self.tmp_shp_y=tf.shape(train_y)
            with tf.Session() as sess:
                out_y = sess.run(self.tmp_shp_y)
                print(out_y)
                self.shape_y.append(None)
                for j in out_y:
                    self.shape_y.append(j)
                print(self.shape_y)
                out = sess.run(self.tmp_shp_x)
                out = out[1:]
                self.shape_x.append(None)
                for x in out:
                    self.shape_x.append(x)
                print(self.shape_x)
                return tf.placeholder(tf.float32,self.shape_x), tf.placeholder(tf.float32,self.shape_y)
        else:
            with tf.Session() as sess:
                out = sess.run(self.tmp_shp_x)
                self.shape_x.append(None)
                for x in out:
                    self.shape_x.append(x)
                print(self.shape_x)
                return tf.placeholder(tf.float32,self.shape_x)


    def model(self):



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
    A = PChild()
    x,y = A.fit(train_x,train_y)
    print("test_initialization")