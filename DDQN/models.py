import tensorflow as tf
import numpy as np
from tensorflow.initializers import random_uniform


class QNet:
    def __init__(self, sess, lr, action_dim, S, S_, tau):
        self.sess = sess
        self.action_dim = action_dim
        self.tau = tau
        self.lr = lr
        self.s = S
        self.s_ = S_

        with tf.variable_scope('QNet'):
            self.q = self._build_net(self.s, 'eval_net', trainable=True)
            self.q_ = self._build_net(self.s_, 'target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='QNet/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='QNet/target_net')

        with tf.variable_scope('target_replace'):
            # self.replace = [tf.assign(t,self.tau*e+(1-self.tau)*t) for t,e in zip(self.t_params,self.e_params)]
            self.replace = [tf.assign(t, e) for t, e in
                            zip(self.t_params, self.e_params)]

    def _build_net(self, s, name, trainable):
        f1 = 1 / 300
        f2 = 0.003
        with tf.variable_scope(name):
            l1 = tf.layers.dense(s, 300, activation=tf.nn.relu, kernel_initializer=random_uniform(-f1, f1),
                                 bias_initializer=random_uniform(-f1, f1), trainable=trainable)

            with tf.variable_scope('q'):
                q = tf.layers.dense(l1, self.action_dim, kernel_initializer=random_uniform(-f2, f2),
                                    bias_initializer=random_uniform(-f2, f2), trainable=trainable)
        return q
