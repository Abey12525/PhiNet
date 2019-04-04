import InitRand
import tensorflow as tf
import numpy as np
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
        InArr = InArr[0]
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
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(logdir='./dense_graph',
                                               graph=sess.graph)
            sess.run(init)
            for num_ne in InArr:
                 for elem in range(num_ne):
                    layer_out_connection  = sess.run(model, feed_dict={x: [[elem]]})
                    mean = np.divide(np.sum(layer_out_connection),output_layer)
                    layer_out_connection = np.where(layer_out_connection >=(mean+(mean*0.04)),1,0)
                    layer_out_connection = np.argwhere(layer_out_connection==1)
                    connection_matrix.append(layer_out_connection)
                    if(elem%200==0):
                        print(len(layer_out_connection))
        print("training_complete !")
        return connection_matrix


if __name__ == "__main__":
    try:
        layers = np.load('./layers.npy')
        Neurons = np.load('./neurons.npy')
        print(layers)
        print(Neurons)
    except:
        print("Error reading File --- !!!!")


    Dynamic = InitRand.RandomInit()
    # function for variable generation
    # tf.reset_default_graph()
    Neuron_number = np.sum(Neurons)
    print(Neuron_number)
    SoftNet = Dense()
    Neurons = Neurons.astype(int)
    Connections = SoftNet.Soft_train(Neurons,Neuron_number)
    np.save('./mask.npy',Connections)
    print("Save Complete !!")