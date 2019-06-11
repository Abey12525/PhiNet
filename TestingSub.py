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






# if __name__=='__main__':
#     """
#     'inr' is used to initialize a random reward , random number of layers and
#     random number of neurons in each layers
#     """
#     lr = Layer()
#     lr.run()
#     print("trial run")

    # #initializing RandomInit class
    # rand_init = inr.RandomInit()
    # #initializing Random Reward --> eg : 10
    # re = rand_init.rnn_reward_rnd_init()
    # #initializing Random number of layers and neuron in each layers --> eg : [250,200,100]
    # layers, neuron_num = rand_init.neuro_rnd_init(np.random.randint(10, 20))
    # #appending reward to the list of number of neurons in each layers --> eg: [250,200,100,10]
    # neuron_num.append(re)
    # #converting to numpy array
    # neuron_num = np.array(neuron_num)
    # #tf.set_random_seed(10)
    # shape = neuron_num.shape[0]
    # """
    # initializing the lstm model
    # passing neuron_num as 3D array since CudnnLSTM is time major
    # """
    # lstm = Layer([[neuron_num]],shape)
    # # lstm_output   - Output of the Model
    # # lstm_state    - State of the Model
    # lstm_output=np.round(lstm.Model())
    # np.save('./layers',lstm_output)
    # print(lstm_output)



# tensorboard --logdir ./path/ --port 7664
#     try:
#         layers = np.load('./layers.npy')
#         Neurons = np.load('./neurons.npy')
#         print(layers)
#         print(Neurons)
#     except:
#         print("Error reading File --- !!!!")
#
#
#     Dynamic = InitRand.RandomInit()
#     # function for variable generation
#     # tf.reset_default_graph()
#     Neuron_number = np.sum(Neurons)
#     print(Neuron_number)
#     SoftNet = Dense()
#     Neurons = Neurons.astype(int)
#     Connections = SoftNet.Soft_train(Neurons,Neuron_number)
#     np.save('./mask.npy',Connections)
#     print("Save Complete !!")

# try:
    #     foo = np.load('./layers.npy')
    #     print(foo)
    # except:
    #     print("File not Found -- !!")
    # Neu = Neurons(int(foo))
    # out = Neu.Model()
    # np.save('./neurons',out)
    # print(out)