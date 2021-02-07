from DQN_agent import DQNAgent
import numpy as np
import gym
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)

EPISODES = 300
BATCH_SIZE = 32
TAU = 0.001
EPSILON = 0.99
GAMMA = 0.97
LR = 1e-3
MEMORY_SIZE = 10000
f1 = 128

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape
action_dim = env.action_space.n

agent = DQNAgent(state_dim=state_dim, action_dim=action_dim,
                 tau=TAU, epsilon=EPSILON, mem_size=MEMORY_SIZE,
                 batch_size=BATCH_SIZE, gamma=GAMMA, lr=LR)
# tf.summary.FileWriter('logs/',agent.sess.graph)

# initialize the buffer with some transitions
counter = 0
while counter < 5 * BATCH_SIZE:
    s = env.reset()
    while True:
        a = agent.choose_action(s)
        s_, r, done, _ = env.step(a)
        agent.store(s, a, r, s_, done)
        counter += 1
        if done:
            break
        s = s_

# start training
all_rewards = []
ori_rewards = []
for ep in range(EPISODES):
    s = env.reset()
    ep_reward = 0
    while True:
        a = agent.choose_action(s)
        s_, r, done, _ = env.step(a)

        ep_reward += r
        agent.store(s, a, r, s_, done)
        agent.learn()
        if done:
            if ep == 0:
                all_rewards.append(ep_reward)
            else:
                all_rewards.append(all_rewards[-1] * 0.9 + ep_reward * 0.1)
            ori_rewards.append(ep_reward)
            break
        s = s_
    print('episode : {}, reward : {}, epsilon : {}'.format(ep, ep_reward, agent.epsilon))

sns.set(style="darkgrid")
x = np.arange(len(all_rewards))
sns.lineplot(x=x, y=all_rewards, legend=False, label=str('DQN'))
plt.title('CartPole-v0')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.legend(loc=4)
plt.show()
