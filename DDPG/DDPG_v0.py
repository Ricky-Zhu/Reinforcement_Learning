import tensorflow as tf
import numpy as np
from utils import OrnsteinUhlenbeckActionNoise
from utils import ReplayBuffer
from models_v2 import Actor,Critic

np.random.seed(1)
tf.set_random_seed(1)

class DDPG:
    def __init__(self,action_dim,action_bound,tau,lr_a,lr_c,state_dim,gamma,batch_size):
        self.target = tf.placeholder(tf.float32,[None,1],'critic_target')
        self.s = tf.placeholder(tf.float32,[None,state_dim],'state')
        self.s_ = tf.placeholder(tf.float32,[None,state_dim],'next_state')

        self.memory = ReplayBuffer(max_size=10000)
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
        self.batch_size = batch_size
        self.gamma = gamma

        self.sess = tf.Session()

        self.actor = Actor(self.sess,self.s,self.s_,action_dim,action_bound,tau,lr_a,f1_units=300)
        self.critic = Critic(self.sess,lr_c,self.s,self.s_,self.actor.a,self.actor.a_,self.target,tau,gamma,state_dim,
                             action_dim,f1_units=300)
        self.actor.add_grad_to_graph(self.critic.a_g)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self,s):
        a = self.actor.choose_action(s)
        var = self.noise()
        a = a+var
        return a[0]

    def update_target_networks(self):
        self.sess.run([self.actor.replace,self.critic.replace])

    def store(self,s,a,r,s_,done):
        self.memory.store(s,a,r,s_,done)

    def learn(self):
        bs,ba,br,bs_, _ = self.memory.sample(self.batch_size)

        q_ = self.sess.run(self.critic.q_,{self.s_:bs_})
        br = br[:,np.newaxis]
        target_critic = br + self.gamma*q_
        self.critic.learn(bs,ba,target_critic)
        self.actor.learn(bs)
        self.update_target_networks()

