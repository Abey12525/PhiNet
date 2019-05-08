import tensorflow as tf
import numpy as np

class dclass:
    def __init__(self):
        we = []
        mask = [[1,0,0],
                [0,1,1]]
        xm = np.array([3,2,4])
        with tf.variable_scope("t",reuse=tf.AUTO_REUSE):
            for i in range(3):
                w = tf.get_variable("w{}".format(i),shape=[2,3],initializer=tf.contrib.layers.xavier_initializer())
                we.append((i,w))
            self.we = dict(we)
        for i in range(3):
            aw = tf.multiply(mask,xm)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            r = sess.run(aw)
        print(r)

if __name__ == '__main__':
    yj = dclass()