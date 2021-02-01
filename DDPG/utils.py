import numpy as np
import random
from collections import deque


class ActionNoise(object):
    def reset(self):
        pass
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma=.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class ReplayBuffer:
    def __init__(self,max_size):
        self.memory = deque(maxlen=max_size)

    def store(self,s,a,r,s_,done):
        self.memory.append((s,a,r,s_,done))

    def length(self):
        return len(self.memory)

    def sample(self,batch_size):
        batchs = random.sample(self.memory,batch_size)
        batch_s, batch_a, batch_r, batch_s_, batch_done = map(np.array,zip(*batchs))
        return batch_s,batch_a,batch_r,batch_s_, batch_done