import torch
import numpy as np
from torchrl.envs import *
from torchrl.envs.libs.gym import *
from IPython.display import clear_output
from torchrl.data import *
import matplotlib.pyplot as plt
from torchrl.data import ReplayBuffer, ListStorage, LazyTensorStorage, LazyMemmapStorage
import logging  
import math
from NetworkArchitecture.NeuralNetwork import NeuralNetwork
from Constants.constants import *
import torch.nn as nn


# want to save best model and current model.

class Agent:
    def __init__(self, num_channels, width, height, n_actions): # initialize Agent
        self.device = self.set_device()
        self.num_channels = num_channels
        self.width = width
        self.height = height
        self.n_actions = n_actions
        self.cnt_frames = 0
        self.initialize_components()

    def set_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_components(self):
        self.setup_networks()
        self.setup_optimizer()
        self.setup_replay_buffer()
        self.setup_agent_properties()

    def setup_networks(self):
        self.policy_network = self.create_network().to(self.device)
        self.target_network = self.create_network().to(self.device)
        self.sync_networks()

    def f(self):
        return (((NUM_FRAMES - self.cnt_frames) / NUM_FRAMES) ** 3) * 0.9 + 0.1

    def get_epsilon(self):
        return max(0.1, self.f())

    def create_network(self):
        return NeuralNetwork(num_channels=self.num_channels, height=self.height, width=self.width,
                             n_actions=self.n_actions)

    def sync_networks(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=LEARNING_RATE)

    def setup_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(storage=LazyMemmapStorage(SIZE))

    def setup_agent_properties(self):
        self.frames = []
        self.loss_history = []
        self.reward_history = []
        self.epsilon = 1
        self.t_max = 10000
        self.reset_counters()

    def reset_counters(self):
        self.count_until_change_target_network = 0

    def select_action(self, state, epsilon=-1):
        if epsilon == -1:
            epsilon = self.get_epsilon()
        if np.random.rand() < epsilon:
            return np.random.choice(np.arange(self.n_actions))
        predicted = self.policy_network(state).detach()
        value, index = torch.max(predicted, 1)
        print(value, index)
        return index.item()
        # s = nn.Softmax(dim=0)
        # probs = s(predicted)
        # # print(predicted)
        # print(probs)
        # action = np.random.choice(np.arange(self.n_actions), p=np.asarray(probs))
        # print("ACTION", action)
        # return action
        # # print(value, index.item())
        # # return index.item()

    def calculate_td_error(self, state, action, reward, next_state, done, gamma=GAMMA):
        with torch.no_grad():
            current_q_value = self.policy_network(state)[0][action]
            next_q_value = self.target_network(next_state).max(1)[0]
            target_q_value = reward + (gamma * next_q_value * (1 - int(done)))
            td_error = abs(target_q_value - current_q_value)
        return td_error

    def compute_loss(self, states, actions, rewards, next_states, dones, gamma=GAMMA):
        states, actions, rewards, next_states, dones = self.convert_to_tensors(states, actions, rewards, next_states, dones)
        current_q_values = self.policy_network(states).gather(1, actions).squeeze(-1)
        next_q_values = self.policy_network(next_states).max(1).values
        expected_q_values = rewards + gamma * (next_q_values * (1 - dones))

        criterion = nn.SmoothL1Loss()
        loss = criterion(current_q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print("loss is", loss.item())
        return loss.item()
    
    def convert_to_tensors(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)  # Convert to long (int) type
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        return states, actions, rewards, next_states, dones

    def record_experience(self, state):
        self.cnt_frames += 1
        self.count_until_change_target_network += 1
        if self.count_until_change_target_network >= FRAMES_FOR_UPDATE_TARGET:
            self.sync_networks()
            self.reset_counters()

        self.replay_buffer.add(state)

    def train(self, batch_size=1000):
        batch = self.replay_buffer.sample(batch_size)
        # print(batch)
        # print(batch['state'])
        # print(batch['action'])
        # print(batch['done'])

        # for i in batch['state']:
        #     plt.imshow(i.permute(1, 2, 0), cmap="gray")
        #     plt.show()
        # print(batch['state'], batch['action'], batch['done'])
        loss = self.compute_loss(batch['state'], batch['action'], batch['reward'], batch['next_state'], batch['done'])
        return loss

    def load(self, filepath='model.pth'):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_networks_and_optimizer(checkpoint)
        self.load_agent_state(checkpoint)

    def load_networks_and_optimizer(self, checkpoint):
        self.policy_network.load_state_dict(checkpoint.get('policy_network', self.create_network().state_dict()))
        self.target_network.load_state_dict(checkpoint.get('target_network', self.create_network().state_dict()))
        self.optimizer.load_state_dict(checkpoint.get('optimizer',torch.optim.Adam(self.policy_network.parameters(), lr=LEARNING_RATE)))

    def load_agent_state(self, checkpoint):
        self.count_until_change_target_network = checkpoint.get('count_until_change_target_network', 0)
        self.frames = checkpoint.get('frames', [])
        self.cnt_frames = checkpoint.get('cnt_frames', 0)
        self.loss_history = checkpoint.get('loss_history', [])
        self.t_max = checkpoint.get('t_max', 10000)
        self.reward_history = checkpoint.get('reward_history', [])
        self.epsilon = checkpoint.get('epsilon', 0.7)

    def save(self, filepath='model.pth'):
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'cnt_frames': self.cnt_frames,
            'count_until_change_target_network': self.count_until_change_target_network,
            'frames': self.frames,
            'loss_history': self.loss_history,
            't_max': self.t_max,
            'reward_history': self.reward_history,
            'epsilon': self.epsilon
        }, filepath)


    def plot_results(self):
        #hide plot
        clear_output(True)
        plt.figure(figsize=(20, 20))
        
        plt.subplot(221)
        plt.title('Rewards per frames')
        plt.xlabel("Frames")
        plt.ylabel("Reward")
        plt.plot(self.frames, self.reward_history, color='blue')
        
        plt.subplot(222)
        plt.title('Loss per frames')
        plt.xlabel("Frames")
        plt.ylabel("Loss")
        plt.plot(self.frames, self.loss_history, color='orange')
        
        plt.subplot(223)
        plt.title('Rewards per last 100 sessions')
        plt.xlabel("Sessions")
        plt.ylabel("Reward")
        plt.plot(self.reward_history[-100:], color='blue')
        
        plt.subplot(224)
        plt.title('Loss per last 100 sessions')
        plt.xlabel("Sessions")
        plt.ylabel("Loss")
        plt.plot(self.loss_history[-100:], color='orange')
        plt.show()