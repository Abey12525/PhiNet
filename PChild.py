import tensorflow as tf
import numpy as np



class PChild():
    def __init__(self):
        try :
            self.neurons = np.load('./neurons.npy')
            self.neuron_mask = np.load('./mask.npy')
            self.layers = len(self.neurons)
            self.neurons = self.neurons.astype(int)
            self.neuron_mask = self.neuron_mask.astype(int)
        except:
            print("Structure Files not Found")
        layers = []
        weights = []
        for i, Neurons in enumerate(self.neurons):
            vars()["w{}".format(i)] = tf.get_variable("W{}".format(i), shape=[Neurons],
                                                      initializer=tf.contib.layers.xavier_initializer())
            vars()["L{}.format(i)"] = []
            str = "w{}".format(i)
            str2 = "L{}".format(i)
            tmp = eval(str)
            tmp2 = eval(str2)
            weights.append(tmp)
            layers.append(tmp2)
        print("layer and weights initialized")
        self.weights, self.layers = weights, layers

    def fit(self,train_x,train_y = None):
        self.train_x = train_x
        self.shape_x = []
        self.tmp_shp_x = tf.shape(train_x)
        if train_y:
            self.shape_y = []
            self.tmp_shp_y=tf.shape(train_y)
            with tf.Session() as sess:
                out_y = sess.run(self.tmp_shp_y)
                self.shape_y.append(None)
                for j in out_y:
                    self.shape_y.append(j)
                print(self.shape_y)
                out = sess.run(self.tmp_shp_y)

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


    def Model(self):
        self.weights,self.layers



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
