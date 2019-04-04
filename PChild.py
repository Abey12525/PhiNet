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


    def fit(self,train_x,train_y = None):
        self.train_x = train_x
        if train_y:
            self.train_y = train_y
        accuracy  = self.Model()
        return accuracy

    def Model(self):

        for x in self.neurons:
            vars()["w{}".format(x)] = tf.get_variable("W{}".format(x),initializer=tf.contib.layers.xavier_initializer())