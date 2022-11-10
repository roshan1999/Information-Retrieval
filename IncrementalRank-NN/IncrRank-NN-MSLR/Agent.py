import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.distributions import Categorical
import time
import numpy as np

# Actor module, categorical actions only
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=0)
        )
    
    def forward(self, X):
        return self.model(X)

# Critic module
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            )
    
    def forward(self, X):
        return self.model(X)


class Agent:
    """Agent class with separate actor and critic networks"""

    def __init__(self, actor_lr, critic_lr, input_dims, gamma=1):
        self.gamma = gamma
        self.actor = Actor(input_dims, 1)
        self.critic = Critic(input_dims)
        self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.score = None
        self.value = None


    def compute_phi(self, critic_state, probs):
        x = np.multiply(critic_state.detach().numpy(), probs.detach().numpy().reshape(probs.shape[0], 1))
        phi_st = x.sum(axis=0)
        phi_st = phi_st.reshape(1, phi_st.shape[0])
        return phi_st

    def choose_action_test(self, observation):
        state = torch.from_numpy(observation).float()

        actor_state = self.actor.forward(state)
        probabilities = actor_state
        probabilities = probabilities.reshape(probabilities.shape[0])
        action = torch.argmax(probabilities)
        return action.item()

    def choose_action(self, observation):
        observation = torch.from_numpy(observation).float()
        actor_state = self.actor.forward(observation)
        probabilities = actor_state
        probabilities = probabilities.reshape(probabilities.shape[0])

        # Getting the chosen action's log probability and storing to update later
        action_probs = Categorical(probabilities)
        action = action_probs.sample()
        self.score = action_probs.log_prob(action)

        phi_s = self.compute_phi(observation, probabilities)
        phi_s = torch.from_numpy(phi_s).float()

        critic_value = self.critic.forward(phi_s)
        value = critic_value.reshape(critic_value.shape[0])
        self.value = critic_value

        return action.item()

    def choose_best_action(self, observation, rel_label):
        observation = torch.from_numpy(observation).float()
        actor_state = self.actor.forward(observation)
        probabilities = actor_state
        probabilities = probabilities.reshape(probabilities.shape[0])

        # Getting the chosen action's log probability and storing to update later
        action_probs = Categorical(probabilities)

        action = torch.tensor(np.argmax(rel_label))
        self.score = action_probs.log_prob(action)

        phi_s = self.compute_phi(observation, probabilities)
        phi_s = torch.from_numpy(phi_s).float()

        critic_value = self.critic.forward(phi_s)
        value = critic_value.reshape(critic_value.shape[0])
        self.value = critic_value

        return action.item()

    def update(self, state, reward, new_state):
        state = torch.from_numpy(state).float()
        new_state = torch.from_numpy(new_state).float()

        done = False
        critic_value_ = 0

        if len(new_state) != 0:
            actor_state = self.actor.forward(new_state)
            probabilities = actor_state
            probabilities = probabilities.reshape(probabilities.shape[0])
            phi_s_ = self.compute_phi(new_state, probabilities)
            phi_s_ = torch.from_numpy(phi_s_).float()
            critic_value_ = self.critic.forward(phi_s_)
            critic_value_ = critic_value_.reshape(critic_value_.shape[0])

        else:
            done = True

        critic_value = self.value

        reward = torch.tensor(reward).reshape(1, 1).float()

        advantage = reward + (1-done) * self.gamma * critic_value_ - critic_value

        critic_loss = advantage.pow(2).mean()
        self.adam_critic.zero_grad()
        critic_loss.backward()
        self.adam_critic.step()

        actor_loss = -self.score * advantage.detach()
        self.adam_actor.zero_grad()
        actor_loss.backward()
        self.adam_actor.step()

        self.score = None
        self.value = None

