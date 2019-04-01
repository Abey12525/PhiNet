import InitRand
import tensorflow as tf
import numpy as np
"""
SoftC
used to initialize n number of variables according to the prediction 
from the RNN
"""
class Dense():
    def Soft_train(self,InArr,output_layer,iteration = 100):
        x = tf.placeholder(tf.float32, [None, 1])
        model = tf.layers.dense(x, units=1500, activation=tf.nn.relu)
        model = tf.layers.dense(model, units=1000, activation=tf.nn.relu)
        model = tf.layers.dense(model, units=800, activation=tf.math.tan)
        model = tf.layers.dense(model, units=900, activation=tf.nn.softmax)
        model = tf.layers.dense(model, units=output_layer, activation=tf.nn.softmax)
        """
        find a function that relates the input and output
        given [3.343,1.235,0.34324,0.23432] input &
        [0.343234,0.34545,0.452,.0234234,.49853495]
        map the input to output 
        """
        # cost = tf.reduce_mean((model-x)**2)
        # opt = tf.train.RMSPropOptimizer(0.01).minimize(cost)
        init = tf.global_variables_initializer()
        # tf.summary.scalar("cost",cost)
        # merge_summary_op  = tf.summary.merge_all()
        for var in InArr:
            with tf.Session() as sess:
                writer = tf.summary.FileWriter(logdir='./dense_graph',
                                               graph=sess.graph)
                sess.run(init)
                InV = sess.run(var[0])
                InV = np.array([np.array(InV)])
                print(InV.shape)
                for elem in InV[0]:
                    for i in range(iteration):
                        result_weight = sess.run(model, feed_dict={x: [[elem]]})
        print("training_complete !")
        return result_weight


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
    Varr,Neuron_number = Dynamic.neuro_rnd_init(20)
    # tf.reset_default_graph()
    Neuron_number = np.sum(Neuron_number)
    SoftNet = Dense()
    Connections = SoftNet.Soft_train(Varr,Neuron_number)
    print(Connections)