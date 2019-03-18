import tensorflow as tf
import numpy as np
import InitRand as sft
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y) = mnist.load_data()
# img = np.array(img)
# img = img.reshape((28,28))
# plt.imshow(img,cmap='gray')
# plt.show()
"""tf.reset_default_graph()"""
#import SoftC as Sft
# inp =tf.constant([[[3562,6,451,512],[15,24,543,457],[563,4642,464,645],[3,23,2,45],[46,575,7,54],[34,56,5,64],[3,65,43,456],[65,34,7,8]]])
# v1 = tf.layers.dense(inp,units=10 ,activation = tf.nn.relu)
# #v2 = tf.layers.dense(v1,units=1024 ,activation = tf.nn.relu )
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(v1))
# var,shp = Sft.SoftCinit()
"""Place holder test"""
sftc = sft.RandomInit()
inp = sftc.neuro_rnd_init()
#inp = tf.Variable([[3,4,5,6]])
# t = tf.add(inp,4)
#inp = tf.get_variable("v1", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
#img = np.reshape(train_x,(60000,784))
shp = np.random.randint(500,784)
img = tf.get_variable("bkn",shape=shp,initializer=tf.contrib.layers.xavier_initializer())
print(img,shp)
for x in inp:
    var = x[0]
    shpa = x[1]
    break
print(var,shpa)
x = tf.placeholder(tf.float32, [None,shpa])
model = tf.layers.dense(x, units=1000,activation=tf.nn.relu)
model = tf.layers.dense(model,units = 500,activation=tf.nn.relu)
model = tf.layers.dense(model,units = 10)
# cost = tf.reduce_mean((model-x)**2)
# opt = tf.train.RMSPropOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()
# tf.summary.scalar("cost",cost)
# merge_summary_op  = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(init)
    img = [np.array(sess.run(var))]
    print(img)
    result_weight = sess.run(model,feed_dict={x : img})
    print(result_weight)