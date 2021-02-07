# Deep Reinforcement Learning

This repository will implement the classic deep reinforcement learning algorithms with regrad to the according papers. The algorithms were implemented in Python with **Tensorflow**. These algorithms will be mainly evaluated in the OpenAI Gym environment.

![TensorFlow](https://img.shields.io/badge/Tensorflow-v1.11.0-green) ![Gym](https://img.shields.io/badge/gym-v%200.17.3-yellow)


## Current Implementation
* Deep Deterministic Policy Gradient (DDPG)  
Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., ... & Wierstra, D. (2015). Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971.

* Deep Q Network (DQN)  
DQN, Double DQN, Dueling DQN were implemented. The norm refers to whether the batch normalization was used.  

The implementation is mainly for the inspiration and focuses on the structure of DQN. The prioritised experience replay (PER) was implemented. But it can just serve a not too big replay buffer size curretly. The PER which can be used for large replay buffer size will be implemented with binary heap data structure in the future.


## Results
### DDPG
<img width="300" src="https://github.com/Ricky-Zhu/Reinforcement_Learning/blob/master/images/ddpg.png"/>

### DQN
<img width="300" src="https://github.com/Ricky-Zhu/Reinforcement_Learning/blob/master/images/comparison_DQN.png"/>


