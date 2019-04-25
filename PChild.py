import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.datasets import mnist

"""
    1. initial condition of weights - Done 
        i) masking condition ? - Done
    2. weights after first initialization - Done
    3. initial condition of layer input x - Done
    4. layer after initial condition - N/A 
    5. mask initialization - Done
    6. mask after initial state -Done
    7.output layer - N/A
"""
class PChild():
    def __init__(self,train_x,train_y=None):
        """Reading the child network structure"""
        try :
            self.neurons = np.load('./neurons.npy')
            self.neurons = self.neurons.astype(int)
            self.neurons = self.neurons[0]
            self.neuron_mask = np.load('./mask.npy')
            self.no_layers = len(self.neurons)
            self.neuron_num = sum(self.neurons)
            self.neurons = self.neurons.astype(int)
            self.neuron_mask = self.neuron_mask.astype(int)
            self.data_len = len(train_x)
            print("data_length: ",self.data_len)
            sys.stdout.write("Structure file found!!!\n")
            sys.stdout.flush()
        except:
            print("Structure Files not Found")

        """creating placeholder for the shape of the input and output"""
        tf.reset_default_graph()
        self.train_x = train_x
        self.shape_x = []
        self.tmp_shp_x = tf.shape(train_x)
        if train_y.any():
            self.shape_y = []
            self.train_y = train_y
            self.tmp_shp_y = tf.shape(train_y)
            with tf.Session() as sess:
                out_y = sess.run(self.tmp_shp_y)
                out_y = out_y[1:]
                self.shape_y.append(None)
                for j in out_y:
                    self.shape_y.append(j)
                out = sess.run(self.tmp_shp_x)
                out = out[1:]
                self.shape_x.append(None)
                for x in out:
                    self.shape_x.append(x)
                self.x_inp,self.y_inp =  tf.placeholder(tf.float32, self.shape_x), tf.placeholder(tf.float32, self.shape_y)
                print("x : ", self.x_inp)
                print("y : ",self.y_inp)
        else:
            print("Unsupervised learning is Currently not supported!!")
            with tf.Session() as sess:
                out = sess.run(self.tmp_shp_x)
                self.shape_x.append(None)
                for x in out:
                    self.shape_x.append(x)
                self.x_inp = tf.placeholder(tf.float32, self.shape_x)
                print("x : ", self.x_inp)

        sys.stdout.write("placeholder initialized\n")
        sys.stdout.flush()
        """Initializing the child network"""
        weights = []
        layers = []
        bias = []
        mask = []
        count = 0

        # Input layer and Input weights
        self.Linp = tf.get_variable("Linp",shape = out,initializer = tf.ones_initializer)
        self.Winp = tf.get_variable("Winp",shape = [self.neurons[0],out],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Binp = tf.get_variable("Binp",shape = [1], initializer=tf.contrib.layers.xavier_initializer())

        for i, Neurons in enumerate(self.neurons):
            vars()["w{}".format(i)] = tf.get_variable("W{}".format(i), shape=[Neurons,self.neuron_num],
                                                      initializer=tf.contrib.layers.xavier_initializer())
            vars()["b{}".format(i)] = tf.get_variable("b{}".format(i), shape=[1],
                                                      initializer = tf.contrib.layers.xavier_initializer())
            vars()["L{}".format(i)] = tf.get_variable("L{}".format(i),shape = [Neurons],
                                                      initializer= tf.ones_initializer)
            vars()["wm{}".format(i)] = self.neuron_mask[count:count+Neurons]
            count += Neurons
            W_str = "w{}".format(i)
            W_tmp = eval(W_str)
            L_str = "L{}".format(i)
            L_tmp = eval(L_str)
            B_str = "b{}".format(i)
            B_tmp = eval(B_str)
            WM_str = "wm{}".format(i)
            WM_tmp = eval(WM_str)
            weights.append(W_tmp)
            layers.append(L_tmp)
            bias.append(B_tmp)
            mask.append(WM_tmp)

        self.Lout = tf.get_variable("Lout", shape = out_y,initializer = tf.ones_initializer)
        self.Wout = tf.get_variable("Wout", shape = [out_y,self.neurons[-1]],
                               initializer = tf.contrib.layers.xavier_initializer())
        self.Bout = tf.get_variable("Bout", shape = [1],initializer=tf.contrib.layers.xavier_initializer())
        self.init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(self.init)
            self.weights = sess.run(weights)
            self.layers = sess.run(layers)
            self.bias = sess.run(bias)
        self.mask = mask
        sys.stdout.write("structure initialization complete\n")
        sys.stdout.flush()



    def model(self,iteration,batch_size=500,learning_rate=0.9):
        self.iteration = iteration
        self.batch_size = batch_size
        self.total_batch = self.data_len // self.batch_size
        print("total_batch : ",self.total_batch)
        print(self.total_batch)
        layer_storage = []
        self.Linp = tf.nn.tanh(tf.add(tf.matmul(self.Winp,self.x_inp), self.Binp))
        for lyr in range(self.no_layers):
            self.mask[lyr] = tf.cast(self.mask[lyr],dtype = tf.float32)
            actual_weights = tf.multiply(self.weights[lyr],self.mask[lyr])   #correct
            layer = tf.concat(self.layers,0)   #concat all layers into one
            actual_layer = tf.multiply(self.mask[lyr], layer)  #correct
            actual_layer_inp = tf.multiply(actual_weights,actual_layer)#check multiplication
            actual_layer_inp = tf.reduce_sum(actual_layer_inp,1)
            layer_result = tf.add(actual_layer_inp,self.bias[lyr]) #correct
            self.layers[lyr] = tf.nn.tanh(layer_result)
        # layer_storage.append(self.layers)
        # self.layers = layer_storage
        self.Lout = tf.multiply(self.Wout,self.layers[-1])
        self.Lout = tf.reduce_sum(self.Lout,1)
        self.Lout = tf.nn.softmax(tf.add(self.Lout,self.Bout))
        with tf.Session() as sess:
            sess.run(self.init)
            for i in range(self.iteration):
                count = 0
                for batch in range(self.total_batch):
                    deepinp = self.train_x[count:count+self.batch_size]
                    batch += 1
                    count +=self.batch_size
                    final_layer = sess.run(self.Lout,feed_dict={self.x_inp : deepinp})
                    print(final_layer)
                    break

    def test_accuracy(self):
        #test accuray using the weights saved
        np.save('./accuracy.npy')




"""
        y_i = x * D_i * W_i = [....x....] * diag([....d....]) * transpose([....w....])
        neurons must be initialized in order to create the network 
        //work in progress 
        for neurons in self.neurons:
            first layer = Neurons
            second layer = Neurons in first layer + Neurons
            third layer = Neurons in second layer + Neurons 
            fourth layer = Neurons in third layer + Neurons 
"""


if __name__ == '__main__':
    (train_x,train_y),(test_x,test_y) = mnist.load_data()
    x = tf.reshape(train_x,[-1,784])
    y = tf.one_hot(test_y,depth = 10)
    with tf.Session() as sess:
        train_x = sess.run(x)
        train_y = sess.run(y)
    child = PChild(train_x,train_y)
    child.model(iteration = 1)
    #print("Accuracy recorded")