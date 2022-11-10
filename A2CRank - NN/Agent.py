import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


class ActorNetwork(nn.Module):

    def __init__(self, actor_lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims  # number of features
        self.fc1_dims = fc1_dims  # number of nodes of layer 1
        self.fc2_dims = fc2_dims  # number of layers
        self.n_actions = n_actions  # 1 (number of documents will stay the same)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)  # Single hidden layer with 45 parameters
        self.pi = nn.Linear(self.fc1_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)

    def forward(self, observation):  # x = features
        """ FeedForward function """
        state = observation
        x = func.relu(self.fc1(state))
        # x = func.relu(self.fc2(x))
        # x = self.fc2(x)  # no approximation
        pi = self.pi(x)
        return pi

class CriticNetwork(nn.Module):

    def __init__(self, critic_lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims  # number of features
        self.fc1_dims = fc1_dims  # number of nodes of layer 1
        self.fc2_dims = fc2_dims  # number of layers
        self.n_actions = n_actions  # 1 (number of documents will stay the same)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)  # Single hidden layer with 45 parameters
        self.v = nn.Linear(self.fc1_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)

    def forward(self, observation):  # x = features
        """ FeedForward function """
        state = observation
        x = func.relu(self.fc1(state))
        # x = func.relu(self.fc2(x))
        # x = self.fc2(x)
        v = self.v(x)
        return v


class Agent:

    def __init__(self, actor_lr, critic_lr, input_dims, gamma=1,
                 layer1_size=46, layer2_size=46, n_actions=1):
        self.gamma = gamma
        self.actor = ActorNetwork(actor_lr, input_dims, layer1_size, layer2_size, n_actions=n_actions)
        self.critic = CriticNetwork(critic_lr, input_dims, layer1_size, layer2_size, n_actions=n_actions)
        self.log_probs = []
        self.state_value = []
        self.rewards = []

    def compute_phi(self, critic_state, probs):
        x = np.multiply(critic_state.detach().numpy(), probs.detach().numpy().reshape(probs.shape[0], 1))
        phi_st = x.sum(axis=0)
        phi_st = phi_st.reshape(1, phi_st.shape[0])

        return phi_st

    def choose_action(self, observation):
        observation = torch.from_numpy(observation).float()
        actor_state = self.actor.forward(observation)
        probabilities = func.softmax(actor_state, dim=0)
        probabilities = probabilities.reshape(probabilities.shape[0])

        # Getting the chosen action's log probability and storing to update later
        action_probs = Categorical(probabilities)
        action = action_probs.sample()
        self.log_probs.append(action_probs.log_prob(action))

        phi_s = self.compute_phi(observation, probabilities)
        phi_s = torch.from_numpy(phi_s).float()

        critic_value = self.critic.forward(phi_s)
        value = critic_value.reshape(critic_value.shape[0])
        self.state_value.append(critic_value)

        return action.item()

    def choose_action_test(self, observation):
        state = torch.from_numpy(observation).float()

        actor_state = self.actor.forward(state)
        probabilities = func.softmax(actor_state, dim=0)
        probabilities = probabilities.reshape(probabilities.shape[0])
        action = torch.argmax(probabilities)
        return action.item()

    def store_values(self, reward, state):

        # Storing the R_{t+1}
        self.rewards.append(reward)

    def update(self):
        actor_loss = None
        critic_loss = None
        # Calculating discounted total return
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)

        # Standardizing the returns
        # returns = (returns - returns.mean() / returns.std())

        loss = 0

        for log_prob, value, R in zip(self.log_probs, self.state_value, returns):
            R = R.reshape(value.shape)
            advantage = R - value.item()

            # Calculate Actor loss
            actor_loss = -log_prob * advantage

            # Calculate Critic loss
            critic_loss = func.smooth_l1_loss(value, R)

            loss += (actor_loss + critic_loss)

        # Update the parameters
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()

        # Reset the rewards, probs, and values
        self.rewards = []
        self.log_probs = []
        self.state_value = []
