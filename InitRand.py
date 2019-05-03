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
        print("----------------------------------------------------")
        print("###### random generation initiated version 2.0 ######")
        print("----------------------------------------------------")

    def neuro_rnd_init(self, rand_num):
        self.vlist = []
        self.No_of_neurons = []
        self.n = rand_num
        tf.reset_default_graph()
        #vars()["b"]=tf.get_variable("b",shape=[self.n],initializer=tf.contrib.layers.xavier_initializer())
        for i in range(self.n):
            self.init_shape = rnd.randint(256, 1024)
            self.No_of_neurons.append(self.init_shape)
            #initializing variables with ordered name ie. v1,v2,...vn
            ##vars()["v".format(i)] is used so that a variable
            ##name can be declared with in the loop
            vars()["v{}".format(i)] = tf.get_variable("v{}".format(i), shape=[self.init_shape],
                                                      initializer=tf.contrib.layers.xavier_initializer())
            #geting the variable by the string name so that it is automaticaly
            #appended to the list of dynamicaly created variables
            str_v ="v{}".format(i)
            tmp = eval(str_v)
            self.vlist.append([tmp,self.init_shape])
        return self.vlist,self.No_of_neurons

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
        return self.norm(reward)


"""
class MyClass(object):
    @staticmethod
    def the_static_method(x):
        print x

MyClass.the_static_method(2) # outputs 2
"""
"""
sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
# We can just use 'c.eval()' without passing 'sess'
print(c.eval())
sess.close()
"""
#v = SoftCinit()
#lst,shp= v.rnd_init()
#print(lst)
#print(shp)
#layers = Dense(lst,shp)

# l = len(lst)
# for i in range(l):
#     print(shp[i])
#
# """
#  vars()["variable_{}".format("name")] = i
#  >>> string = "blah"
# >>> string
# 'blah'
# >>> x = "string"
# >>> eval(x)
# 'blah'
#  """
#
# # #np.save('Indx.npy',[2,3,4])
# # t = np.load('Indx.npy')
# # #x = np.arange(10)
# # #np.save('Indx.npy',np.append(t,x))
# # print(t)
#
# # Create some variables.
# v1 = tf.get_variable("v{}".format(10), shape=[3], initializer=tf.zeros_initializer)
# v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)
#
# inc_v1 = v1.assign(v1 + 1)
# dec_v2 = v2.assign(v2 - 1)
#
# # Add an op to initialize the variables.
# init_op = tf.global_variables_initializer()
#
# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()
#
# # Later, launch the model, initialize the variables, do some work, and save the
# # variables to disk.
# with tf.Session() as sess:
#     sess.run(init_op)
#     # Do some work with the model.
#     inc_v1.op.run()
#     dec_v2.op.run()
#     # Save the variables to disk.
#     save_path = saver.save(sess, "/tmp/model.ckpt")
#     print("Model saved in path: %s" % save_path)
