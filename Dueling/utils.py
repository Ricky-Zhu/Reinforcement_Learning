from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def store(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def length(self):
        return len(self.memory)

    def sample(self, batch_size):
        batchs = random.sample(self.memory, batch_size)
        batch_s, batch_a, batch_r, batch_s_, batch_done = map(np.array, zip(*batchs))
        return batch_s, batch_a, batch_r, batch_s_, batch_done


class PER:
    def __init__(self, max_size, alpha=0, beta=0, epsilon=1e-3):
        self.buffer = deque(maxlen=max_size)
        self.priority = deque(maxlen=max_size)
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta

    def store(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))
        self.priority.append(max(self.priority, default=1))

    def get_sample_probs(self):
        priorities = np.array(self.priority) ** self.alpha
        probs = priorities / np.sum(priorities)
        return probs

    def get_importance(self, probs):
        weights = (1 / (len(self.buffer) * probs)) ** self.beta
        normalized_weights = weights / max(weights)
        return normalized_weights

    def sample(self, batch_size):
        sample_probs = self.get_sample_probs()
        indices = random.choices(np.arange(0, len(self.buffer)), k=batch_size, weights=sample_probs)
        samples = np.array(self.buffer)[indices]
        importance = self.get_importance(sample_probs[indices])

        return map(np.array, zip(*samples)), importance, indices

    def set_priority(self, indices, errors):
        for i, e in zip(indices, errors):
            self.priority[i] = e + self.epsilon
