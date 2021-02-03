import tensorflow as tf
import numpy as np
from models import QNet
from utils import ReplayBuffer

class DQNAgent:
    def __init__(self,state_dim,action_dim,f1,tau,epsilon,mem_size,batch_size,gamma,lr):
        self.sess = tf.Session()

        self.s = tf.placeholder(tf.float32,[None,*state_dim],'state')
        self.s_ = tf.placeholder(tf.float32, [None, *state_dim], 'next_state')
        self.t = tf.placeholder(tf.float32,[None,],'target')
        self.action_in = tf.placeholder(tf.int32,[None,],'action')

        self.action_dim = action_dim
        self.action = tf.one_hot(self.action_in,depth=action_dim)

        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_steps = 5000
        self.decay_inc = (epsilon-0.1)/4000
        self.replace_counter = 0
        self.memory = ReplayBuffer(max_size=mem_size)
        self.Q_net = QNet(sess=self.sess,
                          lr=1e-3,
                          action_dim=action_dim,
                          f1=f1,S=self.s,S_=self.s_,tau=tau)

        self.q_action = tf.reduce_sum(tf.multiply(self.action, self.Q_net.q), axis=1)

        self.error = tf.abs(self.q_action-self.t)
        self.loss = tf.reduce_mean(tf.square(self.error))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


        self.sess.run(tf.global_variables_initializer())

    def choose_action(self,s):
            if s.ndim<2:
                s = [s]
            q_values = self.sess.run(self.Q_net.q,{self.s:s})
            a_best = np.argmax(q_values)
            a = a_best if np.random.random()>self.epsilon else np.random.randint(self.action_dim)
            return a

    def store(self,s,a,r,s_,done):
        self.memory.store(s,a,r,s_,done)

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        q_next = self.sess.run(self.Q_net.q_,{self.s_:next_states})
        q_next[dones] = np.zeros([self.action_dim])
        target = rewards + self.gamma*np.max(q_next,axis=1)

        errors,_ = self.sess.run([self.error,self.train_op],{self.s:states,self.s_:next_states,
                                                             self.action_in:actions,self.t:target
                                                             })
        if self.replace_counter % 300 == 0:
            self.sess.run(self.Q_net.replace)
        self.epsilon = max(0.1,self.epsilon-self.decay_inc)
        self.replace_counter += 1
