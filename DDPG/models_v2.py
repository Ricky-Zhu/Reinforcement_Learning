import tensorflow as tf
import numpy as np
from tensorflow.initializers import random_uniform

class Actor:
    def __init__(self,sess,S,S_,action_dim,action_bound,tau,lr,f1_units):
        self.s = S
        self.s_ = S_
        self.sess = sess
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.tau = tau
        self.lr = lr
        self.f1_units = f1_units

        with tf.variable_scope('Actor'):
            self.a = self._build_net(self.s,'eval_net',trainable=True)
            self.a_ = self._build_net(self.s_,'target_net',trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        with tf.variable_scope('actor_replacement'):
            self.replace = [tf.assign(t,self.tau*e+(1-self.tau)*t) for t,e in zip(self.t_params,self.e_params)]

    def _build_net(self,s,name,trainable):
        f1 = 1 / np.sqrt(400)
        f2 = 1 / np.sqrt(300)
        f3 = 0.003

        with tf.variable_scope(name):

            l1 = tf.layers.dense(s, 400, activation=tf.nn.relu,
                                  kernel_initializer=random_uniform(-f1,f1), bias_initializer=random_uniform(-f1,f1)
                                  , name='l1', trainable=trainable)
            l2 = tf.layers.dense(l1, 300, activation=tf.nn.relu,
                                  kernel_initializer=random_uniform(-f2,f2), bias_initializer=random_uniform(-f2,f2)
                                  , name='l2', trainable=trainable)

            with tf.variable_scope('a'):
                actions = tf.layers.dense(l2, self.action_dim, activation=tf.nn.tanh,
                                          kernel_initializer=random_uniform(-f3,f3),
                                          bias_initializer=random_uniform(-f3,f3), name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound,
                                       name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def choose_action(self,s):
        if s.ndim < 2:
            s = [s]
        return self.sess.run(self.a,{self.s:s})

    def learn(self,s):
        self.sess.run(self.train_op,{self.s:s})


    def add_grad_to_graph(self,a_g):
        with tf.variable_scope('policy_gradient'):
            self.policy_grad = tf.gradients(self.a,self.e_params,a_g)

        self.train_op = tf.train.AdamOptimizer(-self.lr).apply_gradients(zip(self.policy_grad,self.e_params))


class Critic:
    def __init__(self,sess,lr,S,S_,A,A_,T,tau,gamma,state_dim, action_dim,f1_units):
        self.sess = sess
        self.lr = lr
        self.s = S
        self.s_ = S_
        self.a = tf.stop_gradient(A)
        self.a_ = A_
        self.t = T
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.tau = tau
        self.gamma = gamma
        self.f1_units = f1_units

        with tf.variable_scope('Critic'):
            self.q = self._build_net(self.s,self.a,'eval_net',trainable=True)
            self.q_ = self._build_net(self.s_, self.a_, 'target_net', trainable=True)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Critic/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Critic/target_net')

        with tf.variable_scope('critic_replacement'):
            self.replace = [tf.assign(t,self.tau*e+(1-self.tau)*t) for t,e in zip(self.t_params,self.e_params)]

        with tf.variable_scope('action_gradient'):
            self.a_g = tf.gradients(self.q,self.a)[0]

        self.loss = tf.reduce_mean(tf.squared_difference(self.t, self.q))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _build_net(self,s,a,name,trainable):
        f1 = 1/np.sqrt(400)
        f2 = 1 / np.sqrt(300)
        f3 = 0.003

        with tf.variable_scope(name):
            with tf.variable_scope('l1'):
                l1 = tf.layers.dense(s,400,kernel_initializer=random_uniform(-f1,f1),
                                     bias_initializer=random_uniform(-f1,f1),trainable=trainable)
                l2 = tf.layers.dense(l1,300,kernel_initializer=random_uniform(-f2,f2),
                                     bias_initializer=random_uniform(-f2,f2),trainable=trainable)
                l2 = tf.layers.batch_normalization(l2)
                a_in = tf.layers.dense(a,300,kernel_initializer=random_uniform(-f2,f2),
                                       bias_initializer=random_uniform(-f2,f2),trainable=trainable)

                net = tf.nn.relu(l2+a_in)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=random_uniform(-f3,f3),
                                    bias_initializer=random_uniform(-f3,f3),
                                    trainable=trainable)  # Q(s,a)
        return q

    def learn(self,s,a,t):
        self.sess.run(self.train_op,{self.s:s,self.a:a,self.t:t})