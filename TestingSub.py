import tensorflow as tf
import numpy as np

class test:

    def __init__(self):
        self.layer_len = 4
        self.weights = []
        tf.reset_default_graph()
        with tf.variable_scope("re",reuse=tf.AUTO_REUSE):
            self.mask = tf.get_variable("msk",shape = [10,10],initializer = tf.zeros_initializer)
            for i in range(self.layer_len):
                vars()["w125{}".format(i)]  = tf.get_variable("w125{}".format(i),shape = [10,10],
                                                              initializer = tf.contrib.layers.xavier_initializer())
                str = "w125{}".format(i)
                tmp = eval(str)
                self.weights.append(tmp)


    # def sub(self):
    #     i = tf.constant(100)
    #     self.lyr = 0
    #     li = self.layer_len
    #     def condition(i,li):
    #         return tf.less(i,tf.constant(li))
    #
    #     def body(i,li):
    #         actual_weights = tf.multiply(self.weights[i], self.mask)
    #         i = tf.add(i,1)
    #         return i
    #     i,Loop = tf.while_loop(condition, body, i,li)
    #
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         sess.run(i)
    #         L =sess.run(Loop)
    #         print(L)
    #     print("test run")

if __name__ == '__main__':
    k = test()
    k.sub()






















#
# mask = np.load('./mask.npy')
# mask_o = mask[0:2]
#
# tf.reset_default_graph()
# with tf.variable_scope("test343",reuse= tf.AUTO_REUSE):
#     w = tf.get_variable("winp4", dtype=tf.float32, shape=(500, 784),
#                                 initializer=tf.contrib.layers.xavier_initializer())
#     test = tf.get_variable("inittest", shape=(784), initializer=tf.contrib.layers.xavier_initializer())
#
# mul = tf.multiply(w,test)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     j = sess.run(mul)
#     print(j.shape)


# class test:
#     def __init__(self):
#         self.t = 10
#         self.k =5
#         self.a = [3,8,5]
#
#     def tt(self):
#         def test(i):
#             self.t =40
#             self.k=10
#             print(self.a[i])
#             self.a[i]=0
#         for i in range(3):
#             test(i)
#         print(self.t)
#         print(self.k+5)
#         print(self.a)
#
# if __name__=='__main__':
#     k = test()
#     k.tt()



