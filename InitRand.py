"""
step1 : return dynamic number of tensors and it's shape
step2 : return the tensors and shape to Dense class
step3 : use both the shape and tensors for creating the
        dense network
sept4 : initialize and train
"""
import numpy as np
import tensorflow as tf
import random as rnd

class normal():
    def norm(self,In_x):
        #normalizing value to -5 to 5
        return ((In_x)*0.1)-5.0


class RandomInit(normal):
    def __init__(self):
        print("-----------------------------------------")
        print("###### random generation initiated ######")
        print("-----------------------------------------")

    def neuro_rnd_init(self, rand_num):
        self.vlist = []
        self.No_of_neurons = []
        self.n = rand_num
        tf.reset_default_graph()
        #vars()["b"]=tf.get_variable("b",shape=[self.n],initializer=tf.contrib.layers.xavier_initializer())
        self.no_of_neurons = np.random.randint(256,1024,(self.n+1))
        print("+++++++")
        print("Sample Number of neurons per layer",self.no_of_neurons)
        print("+++++++")
        for i,elem in enumerate(self.no_of_neurons):
            # self.init_shape = rnd.randint(256, 1024)
            # self.No_of_neurons.append(self.init_shape)
            #initializing variables with ordered name ie. v1,v2,...vn
            ##vars()["v".format(i)] is used so that a variable
            ##name can be declared with in the loop
            vars()["v{}".format(i)] = tf.get_variable("v{}".format(i), shape=[elem],
                                                      initializer=tf.contrib.layers.xavier_initializer())
            #geting the variable by the string name so that it is automaticaly
            #appended to the list of dynamicaly created variables
            str_v ="v{}".format(i)
            tmp = eval(str_v)
            self.vlist.append([tmp,elem])
        print("variable created :",self.vlist)
        print("++++++")
        return self.vlist,self.no_of_neurons

    def rnn_reward_rnd_init(self):
        """reward is normalized -5 to 5
        the accuracy will be between 0% to 100%
        normalization equation == X' = (b-a) X  - minX + a
                                            ----------
                                            maxX - minX
        """
        # with tf.variable_scope("rnn_reward_rnd_init", reuse=tf.AUTO_REUSE):
        #     reward = tf.get_variable("reward3",initializer = tf.random_uniform([1],0,100))
        #     re = self.norm(reward)
        # print(re)
        reward = np.random.randint(50, 100)
        np.save('./reward.npy',reward)
        return self.norm(reward)

if __name__ == '__main__':
    test_gen = RandomInit()
    vlist,no_of_neurons = test_gen.neuro_rnd_init(np.random.randint(3,8))



