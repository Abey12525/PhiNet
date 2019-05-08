import numpy as np
import tensorflow as tf

class Pchpro:
    def __init__(self):
        self.pro_lyr = np.load('./pro_lyr.npy')
        self.neurons = np.load('./neurons.npy')
        self.neurons = self.neurons.astype(int)
        self.neurons = self.neurons[0]
        self.neuron_mask = np.load('./mask.npy')
        self.no_layers = len(self.neurons)
        self.neuron_num = sum(self.neurons)
        self.neurons = self.neurons.astype(int)
        self.neuron_mask = self.neuron_mask.astype(int)

    def test(self):
        weights = []
        bias = []
        mask = []
        with tf.variable_scope("pro",reuse=tf.AUTO_REUSE):
            self.Winp = tf.get_variable("Winp", shape=[784, self.neurons[0]],
                                        initializer=tf.contrib.layers.xavier_initializer())
            self.Binp = tf.get_variable("Binp", shape=[1], initializer=tf.contrib.layers.xavier_initializer())

            count = 0
            for i, Neurons in enumerate(self.neurons):
                vars()["w{}".format(i)] = tf.get_variable("W{}".format(i), shape=[Neurons, self.neuron_num],
                                                          initializer=tf.contrib.layers.xavier_initializer())
                vars()["b{}".format(i)] = tf.get_variable("b{}".format(i), shape=[1],
                                                          initializer=tf.contrib.layers.xavier_initializer())
                vars()["l{}".format(i)] = tf.get_variable("l{}".format(i), shape=[Neurons], initializer=tf.ones_initializer)
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

            self.Wout = tf.get_variable("Wout", shape=[self.neurons[-1], 10],
                                        initializer=tf.contrib.layers.xavier_initializer())
            self.Bout = tf.get_variable("Bout", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
        self.weights = dict(weights)
        self.bias = dict(bias)
        self.mask = mask
        print("initialization")
        actual_weights = tf.placeholder(tf.float32,shape=[])
        aw_str = []
        for lyr in range(self.no_layers):
            act_we = tf.multiply(self.weights[lyr],self.mask[lyr])
            aw_str.append(act_we)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.a_w = sess.run(aw_str)
        ly_r = tf.placeholder(tf.float32,shape=[None,self.neurons[-1]])
        res = tf.matmul(ly_r, self.Wout)
        res = tf.add(res,self.Bout)
        #res = tf.argmax(res,axis=1)
        for i, pr_inp in enumerate(self.pro_lyr):
            print("processed_array : ")
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for lyr in range(self.no_layers):
                    actu_l = sess.run(tf.multiply(self.mask[lyr],pr_inp))
                    result = sess.run(tf.multiply(self.a_w[lyr],actu_l))
                    ac_lyr = sess.run(tf.reduce_sum(result,1))
                    lyr_res = sess.run(tf.add(ac_lyr,self.bias[lyr]))
                    self.lyr_res2 = sess.run(tf.nn.tanh(lyr_res))
                result_argmax = sess.run(res,feed_dict={ly_r : [self.lyr_res2]})
                res3 = sess.run(tf.argmax(result_argmax,axis=1))
                print("#######################################################")
                print(result_argmax)
                print(res3)
                print("#######################################################")
                if i == 2:
                    break
                sess.close()








if __name__ == '__main__':
    x = Pchpro()
    x.test()







# print(np.shape(lyr_xcpy))
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#             for pre_pro in lyr_xcpy:
#                 actual_weights = sess.run(tf.multiply(self.weights[lyr], self.mask[lyr]))
#                 actual_layer = sess.run(tf.multiply(self.mask[lyr], pre_pro))  # correct
#                 actual_layer_inp = sess.run(tf.multiply(actual_weights, actual_layer))  # check multiplication
#                 actual_layer_inp = sess.run(tf.reduce_sum(actual_layer_inp, 1))
#                 layer_result = sess.run(tf.add(actual_layer_inp, self.bias[lyr]))  # correct
#                 layer_result = sess.run(tf.nn.tanh(layer_result))
#                 print(layer_result)