import tensorflow as tf
import numpy as np



class PChild():
    def __init__(self):
        try :
            self.neurons = np.load('./neurons.npy')
            self.neuron_mask = np.load('./mask.npy')
            self.layers = len(self.neurons)
        except:
            print("Structure Files not Found")


    def fit(self,train_x,train_y = None):
        self.train_x = train_x
        if train_y:
            self.train_y = train_y
            print("Unsupervised Learning ?? Hmmm. . . ")
        accuracy  = self.Model()
        return accuracy 

    def Model(self):
