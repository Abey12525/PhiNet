import tensorflow as tf
import time


#timestr = time.strftime("%Y%m%d-%H%M%S")
tf.reset_default_graph
#saver = tf.train.Saver()
class Layers():

    def LSTM():
        Lstm_1 = tf.contrib.cudnn_rnn.CudnnLSTM(10,5)
        output1, state1 = Lstm_1(x,initial_state = None, training = True)
        Lstm_2 = tf.contrib.cudnn_rnn.CudnnLSTM(10,4)
        output2,state2 = Lstm_2(output,initial_state = state1, training = True)
        Lstm_3 = tf.contrib.cudnn_rnn.CudnnLSTM(6,3)
        output3,state3 = Lstm_3(out,initial_state = state2, training = True)
        Lstm_4 = tf.contrib.cudnn_rnn.CudnnLSTM(10,1)
        Model_out,Model_state = Lstm_4(out1,initial_state = None,training = True)
        return Model_out,Model_state
    """
    tf.contrib.layers.fully_connected(
        inputs,
        num_outputs,
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None
    )
    """
if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32,shape=[None, 1, 1])
    lstm_out,lstm_state = Layers.LSTM()
    with tf.Session() as sess:
        #Writer = tf.summary.FileWriter(logdir='./graphs/G{}/'.format(timestr),graph=sess.graph)
        sess.run(tf.global_variables_initializer())
        x = sess.run(lstm_out,feed_dict={x : [[[20]]]})
        #save_path = saver.save(sess, "./tmp/model.ckpt")
        #print("Model saved in path: %s" % save_path)


