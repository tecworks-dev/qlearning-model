import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Optional
import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.alpha = alpha

    def add(self, experience, priority=1.0):
        max_priority = max(self.priorities, default=priority)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[:self.position])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return experiences, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class QLearningAgent:
    def __init__(self, model, config, trainloader):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'], weight_decay=config['train']['weight_decay'])
        self.scheduler = OneCycleLR(self.optimizer, max_lr=config['train']['max_lr'], steps_per_epoch=len(trainloader), epochs=config['train']['epochs'])
        self.gamma = config['train']['gamma']
        self.buffer = PrioritizedReplayBuffer(config['train']['buffer_size'])
        self.batch_size = config['train']['batch_size']
        self.scaler = GradScaler()

    def update(self, state, action, reward, next_state, done):
        self.buffer.add((state, action, reward, next_state, done))

        if len(self.buffer.buffer) < self.batch_size:
            return

        experiences, indices, weights = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.stack(rewards).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.stack(dones).to(device)

        with autocast():
            state_action_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            next_state_values = self.model(next_states).max(1)[0].detach()
            expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))

            loss = F.mse_loss(state_action_values, expected_state_action_values, reduction='none')
            loss = (loss * torch.tensor(weights).to(device)).mean()

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.scheduler.step()
