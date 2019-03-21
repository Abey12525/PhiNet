import tensorflow as tf
print(tf.__version__)
import InitRand

# class Ne:
#     init_scale = 0.1
#
#
# flags = tf.flags
#
# flags.DEFINE_string("model","small","testing")
#
#
#
# FLAGS = flags.FLAGS
# print("model is :",FLAGS.model)
#
# if FLAGS.model == "small":
#     print("test phase_one complete :",FLAGS.model)
#     obj = Ne
#
#
# def test(model):
#     print(model.init_scale)
#
# test(obj)


x = InitRand.RandomInit()
print(x.rnn_reward_rnd_init())