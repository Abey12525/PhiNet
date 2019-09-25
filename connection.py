import InitRand
import tensorflow as tf
import numpy as np
import Neuron_network as nn
import sys
"""
SoftC
used to initialize n number of variables according to the prediction 
from the RNN
"""
""""
Dropout
__init__(
    rate=0.5,
    noise_shape=None,
    seed=None,
    name=None,
    **kwargs
)
"""
class Dense():
    def Soft_train(self,InArr,output_layer,iteration = 1):
        connection_matrix = []
        x = tf.placeholder(tf.float32, [None, 1])
        model = tf.layers.dense(x, units=1500, activation=tf.nn.relu,bias_initializer=tf.contrib.layers.xavier_initializer())
        model = tf.layers.dense(model, units=1000, activation=tf.nn.relu,bias_initializer=tf.contrib.layers.xavier_initializer())
        model = tf.layers.dropout(model,rate = 0.5)
        model = tf.layers.dense(model, units=800, activation=tf.math.tan,bias_initializer=tf.contrib.layers.xavier_initializer())
        model = tf.layers.dropout(model,rate = 0.5)
        model = tf.layers.dense(model, units=900, activation=tf.nn.softmax,bias_initializer=tf.contrib.layers.xavier_initializer())
        model = tf.layers.dense(model, units=output_layer, activation=tf.nn.softmax,bias_initializer=tf.contrib.layers.xavier_initializer())
        """
        find a function that relates the input and output
        given [3.343,1.235,0.34324,0.23432] input &
        [0.343234,0.34545,0.452,.0234234,.49853495]
        map the input to output 
        """
        """
        Insted of predicting values for each neurons predict a connection matrix
        [[ni,ni+10,ni+20,]      ,[nj]               ,[nk,nk+1]      ]
        . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        [[nm,nm+1,nm+4,nm+5]    ,[nb+76,nb+10,nb+2] ,[nl+9,nl+3]    ]
        """
        # cost = tf.reduce_mean((model-x)**2)
        # opt = tf.train.RMSPropOptimizer(0.01).minimize(cost)
        init = tf.global_variables_initializer()
        # tf.summary.scalar("cost",cost)
        # merge_summary_op  = tf.summary.merge_all()
        print("Training")
        sys.stdout.write("||")
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(logdir='./dense_graph',
                                               graph=sess.graph)
            sess.run(init)
            for num_ne in InArr:
                 for elem in range(num_ne):
                    layer_out_connection  = sess.run(model, feed_dict={x: [[elem]]})
                    mean = np.divide(np.sum(layer_out_connection),output_layer)
                    layer_out_connection = np.where(layer_out_connection >=(mean+(mean*0.004)),1,0)
                    connection_matrix.append(layer_out_connection[0])
                    if(elem%200==0):
                        sys.stdout.write("#")
                        sys.stdout.flush()
        print("||\ntraining_complete !")
        print(np.shape(connection_matrix))
        return connection_matrix

if __name__ == "__main__":
    connection = Dense()
    array_example = [100,120,300]
    out_array = np.sum(array_example)
    connection_matrix = connection.Soft_train(array_example,out_array)
    print(connection_matrix)
    print(" Neuro_connection found ")




