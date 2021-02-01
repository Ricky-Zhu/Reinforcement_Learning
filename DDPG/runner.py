from DDPG_v0 import DDPG
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

LR_A = 1e-4    # learning rate for actor
LR_C = 1e-3    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.001      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
env = gym.make('Pendulum-v0')


agent = DDPG(action_dim=1,action_bound=env.action_space.high,tau=TAU,lr_a=LR_A,lr_c=LR_C,
             state_dim=env.observation_space.shape[0],gamma=GAMMA,batch_size=BATCH_SIZE)

# tf.summary.FileWriter("logs/", agent.sess.graph)

EPISODES = 500
initilization_done = False
all_rewards = []
for ep in range(EPISODES):
    s = env.reset()
    agent.noise.reset()
    ep_reward = 0
    while True:
        a = agent.choose_action(s)
        s_,r,done,_ = env.step(a)
        ep_reward += r
        agent.store(s,a,r,s_,done)
        if initilization_done:
            agent.learn()
        if done:
            all_rewards.append(ep_reward)
            break
        s = s_
    if agent.memory.length() > 100:
        initilization_done = True

    print('ep: {}, reward: {}'.format(ep,ep_reward))


sns.set(style="darkgrid")
x = np.arange(len(all_rewards))
sns.lineplot(x=x,y=all_rewards)
plt.show()
