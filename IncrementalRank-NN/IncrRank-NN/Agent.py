import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.distributions import Categorical
import time
import numpy as np

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
            nn.Softmax(dim=0)
        )
    
    def forward(self, X):
        return self.model(X)

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, X):
        return self.model(X)


"""Agent class initialized with separate actor and critic networks"""
class Agent:

    def __init__(self, actor_lr, critic_lr, input_dims, gamma=1):
        self.gamma = gamma
        self.actor = Actor(input_dims, 1)
        self.critic = Critic(input_dims)

        # Initializing the learning rates of actor and critic
        self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.score = None
        self.value = None


    # Computing phi(st) for the value function
    def compute_phi(self, critic_state, probs):
        x = np.multiply(critic_state.detach().numpy(), probs.detach().numpy().reshape(probs.shape[0], 1))
        phi_st = x.sum(axis=0)
        phi_st = phi_st.reshape(1, phi_st.shape[0])
        return phi_st

    # Selecting greedy action while performing testing
    def choose_action_test(self, observation):
        state = torch.from_numpy(observation).float()

        # Passing the current state to the neural network
        actor_state = self.actor.forward(state)

        # Getting the probabilities provided by softmax
        probabilities = actor_state
        probabilities = probabilities.reshape(probabilities.shape[0])
        
        # Choosing the greedy action for testing
        action = torch.argmax(probabilities)
        return action.item()

    # Selection action based on categorical distribution
    def choose_action(self, observation):
        observation = torch.from_numpy(observation).float()

        # Passing the current state to the neural network
        actor_state = self.actor.forward(observation)
        
        # Getting the probabilities provided by softmax
        probabilities = actor_state
        probabilities = probabilities.reshape(probabilities.shape[0])

        # Getting the chosen action's score function and storing to use for update later
        action_probs = Categorical(probabilities)
        action = action_probs.sample()
        self.score = action_probs.log_prob(action)

        # Calculating the state for the critic, using probabilities of the actor
        phi_s = self.compute_phi(observation, probabilities)
        phi_s = torch.from_numpy(phi_s).float()

        # Computing the critic's value function and storing to use for update later
        critic_value = self.critic.forward(phi_s)
        value = critic_value.reshape(critic_value.shape[0])
        self.value = critic_value

        return action.item()


    # Selecting optimal action
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


    # Updating the actor and critic parameters
    def update(self, state, reward, new_state):
        # state for the value function of the current state
        state = torch.from_numpy(state).float()

        # new_state for the value function of the next state
        new_state = torch.from_numpy(new_state).float()

        # Indicates if trajectory has ended
        done = False
        critic_value_ = 0

        # If not the last state in the trajectory compute the 
        # next state's value function
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

        # Fetching current state's critic value from buffer
        critic_value = self.value

        # Converting reward to tensor (to use in-built functions)
        reward = torch.tensor(reward).reshape(1, 1).float()

        # Using the done flag to omit the value of next state when not present
        advantage = reward + (1-done) * self.gamma * critic_value_ - critic_value

        # Calculate the critic's loss
        critic_loss = advantage.pow(2).mean()

        # Reset the gradients of the critic
        self.adam_critic.zero_grad()

        # Backpropagate the critic loss
        critic_loss.backward()
        self.adam_critic.step()

        # Calculate the actor's loss
        actor_loss = -self.score * advantage.detach()

        # Reset the gradient of the actor
        self.adam_actor.zero_grad()

        # Backpropagate the actor loss
        actor_loss.backward()
        self.adam_actor.step()

        # Reset the buffers
        self.score = None
        self.value = None

