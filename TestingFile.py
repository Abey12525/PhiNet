import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

class test:
    def __init__(self,train_x,train_y):
        try:
            self.neurons = np.load('./neurons.npy')
            self.neurons = self.neurons.astype(int)
            self.neuron_mask = np.load('./mask.npy')
            self.layers = len(self.neurons[0])
        except:
            print("Structure file not Found !!!")

        self.neurons = self.neurons[0]
        self.t_neurons = sum(self.neurons)
        w = []
        layer = []
        mask = []
        m_count = 0
        tf.reset_default_graph()
        with tf.variable_scope("test",reuse=tf.AUTO_REUSE):
            self.x = tf.placeholder(tf.float32,shape = [None,784])
            self.y = tf.placeholder(tf.float32,shape = [None, 10])
            self.Linp = tf.get_variable("Linp3",dtype=tf.float32,shape = [784], initializer = tf.ones_initializer())
            self.winp = tf.get_variable("winp3",dtype= tf.float32,shape = (self.neurons[0],784),
                                   initializer = tf.contrib.layers.xavier_initializer())

            for i,neurons in enumerate(self.neurons):
                vars()["w{}".format(i)] = tf.get_variable("wlk{}".format(i),shape = [neurons,self.t_neurons],
                                                          initializer = tf.contrib.layers.xavier_initializer())
                vars()["L{}".format(i)] = tf.get_variable("Lki{}".format(i),shape = [neurons],
                                                          initializer = tf.ones_initializer,trainable = False)
                vars()["M{}".format(i)] = self.neuron_mask[m_count:m_count+neurons]

                str1 = "w{}".format(i)
                str2 = "L{}".format(i)
                str3 = "M{}".format(i)
                tmp1 = eval(str1)
                tmp2 = eval(str2)
                tmp3 = eval(str3)
                w.append(tmp1)
                layer.append(tmp2)
                mask.append(tmp3)
            self.Lout = tf.get_variable("Lout3",shape = [10], initializer = tf.ones_initializer())
            self.wout = tf.get_variable("wout3",shape = [10,self.neurons[-1]])
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            self.weights = sess.run(w)
            self.layers = sess.run(layer)
        self.mask = mask
        print("++++++++++++++init++++++++++++++")


    def train(self):
        print("__train__")
        with tf.variable_scope("train",reuse = tf.AUTO_REUSE):
            test = tf.get_variable("inittest",shape = (784),initializer = tf.contrib.layers.xavier_initializer())
        mask_t = tf.cast(self.mask[0],dtype=tf.float32)
        actual_layer = tf.multiply(self.winp,test)
        self.Linp = actual_layer
        actual_layer = tf.matmul(self.Linp,mask_t)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            y = sess.run(actual_layer)
            t = sess.run(mask_t)
            tm = sess.run(self.Linp)
            wi = sess.run(self.winp)
            print("-----inp_weight----")
            print(wi.shape)
            print("----mask-----")
            print(mask_t.shape)
            print("-----winp*784----")
            print(tm.shape)
            print("----actual_layer-----")
            print(y.shape)
            print("the actual_dim ")

if __name__ == '__main__':
    (train_x,train_y),(test_x,test_y) = mnist.load_data()
    t = test(train_x,test_y)
    t.train()











# def data_gen():
#     tf.reset_default_graph()
#     k = np.random.uniform(2,15)
#     k = int(np.round(k))
#     var_list = []
#     rw_list = []
#     for i in range(k):
#         x = np.random.uniform(0, 100)
#         rw_list.append(x)
#         vars()["vk{}".format(i)] = tf.get_variable("vk{}".format(i),shape = [int(np.round(np.random.uniform(200,500)))],dtype = tf.float32 ,initializer =tf.contrib.layers.xavier_initializer())
#         tmp = eval("vk{}".format(i))
#         var_list.append(tmp)
#     var_list.append(x)
#     return var_list,rw_list

# def Model(data,rw):
#     x = tf.placeholder(dtype = tf.float32, shape=[None,1,1])
#     model = tf.contrib.cudnn_rnn.CudnnLSTM(10,5)
#     out,state = model(x,initial_state = None, training = True)
#     rw = tf.Variable([-5],dtype=tf.float32)
#     opt = tf.train.GradientDescentOptimizer(out).minimize(rw)
#     init = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init)
#         for i in range(100):
#             x = sess.run(opt,feed_dict={x : [[[data]]]})
#             if(i%25 == 0):
#                 print("optimizing : ",x)


# from tensorflow.keras.datasets import mnist
# (train_x,train_y),(test_x,test_y) = mnist.load_data()
#
# x = tf.shape(train_x)
# with tf.Session() as sess:
#     x = sess.run(x)
#     y = []
#     z = x[1:].astype(int)
#     z = z[:]
#     y.append(None)
#     for j in z:
#         print(j)
#         y.append(j)

# rw = [[[0],[4],[5],[-5]]]
# Model([[[3,5,2,7],[4,5,6,8],[1,9,2,5],[4,6,8,7]]],rw)


#
# import time
# import sys
#
# toolbar_width = 40
#
# # setup toolbar
# sys.stdout.write("[%s]" % (" " * toolbar_width))
# sys.stdout.flush()
# sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
#
# for i in range(toolbar_width):
#     time.sleep(0.1) # do real work here
#     # update the bar
#     sys.stdout.write("-")
#     sys.stdout.flush()
#
# sys.stdout.write("\n")