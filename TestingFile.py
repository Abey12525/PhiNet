import tensorflow as tf
import numpy as np

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


from tensorflow.keras.datasets import mnist
(train_x,train_y),(test_x,test_y) = mnist.load_data()

x = tf.shape(train_x)
with tf.Session() as sess:
    x = sess.run(x)
    y = []
    z = x[1:].astype(int)
    z = z[:]
    y.append(None)
    for j in z:
        print(j)
        y.append(j)

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