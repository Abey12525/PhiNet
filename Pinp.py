import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

class ChModel:
    def __init__(self,train_x,train_y):
        enc = OneHotEncoder()
        train_y = train_y.reshape(-1, 1)
        enc.fit(train_y)
        train_y = enc.transform(train_y).toarray()
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

        tf.reset_default_graph()
        self.train_x = train_x / 255
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
        print(self.shape_y,self.shape_x)
        weights = []
        bias = []
        mask = []
        layer = []
        tf.reset_default_graph()

        self.Linp = tf.get_variable("Linp", shape=out, initializer=tf.ones_initializer)
        self.Winp = tf.get_variable("Winp", shape=[out, self.neurons[0]],
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.Binp = tf.get_variable("Binp", shape=[1], initializer=tf.contrib.layers.xavier_initializer())

        count = 0
        for i, Neurons in enumerate(self.neurons):
            vars()["w{}".format(i)] = tf.get_variable("W{}".format(i), shape=[Neurons, self.neuron_num],
                                                      initializer=tf.contrib.layers.xavier_initializer())
            vars()["b{}".format(i)] = tf.get_variable("b{}".format(i), shape=[1],
                                                      initializer=tf.contrib.layers.xavier_initializer())
            vars()["l{}".format(i)] = tf.get_variable("l{}".format(i),shape=[Neurons], initializer=tf.ones_initializer)
            vars()["wm{}".format(i)] = self.neuron_mask[count:count + Neurons]
            count += Neurons
            W_str = "w{}".format(i)
            W_tmp = eval(W_str)
            B_str = "b{}".format(i)
            B_tmp = eval(B_str)
            l_str = "l{}".format(i)
            l_tmp = eval(l_str)
            WM_str = "wm{}".format(i)
            WM_tmp = eval(WM_str)
            weights.append((i, W_tmp))
            bias.append((i, B_tmp))
            mask.append(WM_tmp)

        self.Wout = tf.get_variable("Wout", shape=[self.neurons[-1], out_y],
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.Bout = tf.get_variable("Bout", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
        self.init = tf.global_variables_initializer()
        self.weights = dict(weights)
        self.bias = dict(bias)
        self.mask = mask
        layers = []
        for lyr in self.neurons:
            tmp = np.ones(lyr)
            layers.append(tmp)
        self.layers = layers
        print("initialization")

    def model(self):
        x_inp = tf.placeholder(tf.float32, self.shape_x)
        y_inp = self.y_inp = tf.placeholder(tf.float32, self.shape_y)
        ly_r = tf.placeholder(tf.float32,[None,self.neurons[-1]])
        va = tf.matmul(x_inp,self.Winp)
        va = tf.nn.tanh(tf.add(va,self.Binp))
        res = tf.matmul(ly_r,self.Wout)
        res = tf.add(res,self.Bout)
        init = tf.global_variables_initializer()
        self.processed_inp = []
        layers = []
        inp_str = []
        with tf.Session() as sess:
            sess.run(init)
            for i,x in enumerate(self.train_x):
                var = sess.run(va, feed_dict={x_inp : [x]})
                self.processed_inp.append(var)
            for i,p_inp in enumerate(self.processed_inp):
                lyr_cpy = self.layers
                lyr_cpy[0] = p_inp
                for lyr in range(self.no_layers):
                    lyr_i = []
                    for li in range(self.no_layers):
                        l = lyr_cpy[li]
                        lyr_i = np.append(lyr_i,l)
                    actual_weights = sess.run(tf.multiply(self.weights[lyr], self.mask[lyr]))
                    actual_layer = sess.run(tf.multiply(self.mask[lyr], lyr_i))  # correct
                    actual_layer_inp = sess.run(tf.multiply(actual_weights, actual_layer))  # check multiplication
                    actual_layer_inp = sess.run(tf.reduce_sum(actual_layer_inp, 1))
                    layer_result = sess.run(tf.add(actual_layer_inp, self.bias[lyr]))  # correct
                    layer_result = sess.run(tf.nn.tanh(layer_result))
                    lyr_cpy[lyr] = layer_result
                #res = sess.run(res, feed_dict={ly_r : [layer_result]})
                print(layer_result)
                if i == 4 :
                    break








if __name__ == '__main__':
    (train_x,train_y),(test_x,test_y) = mnist.load_data()
    train_x = np.reshape(train_x,[-1,784])
    child = ChModel(train_x,train_y)
    child.model()
    #print("Accuracy recorded")