import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

T.manual_seed(7)


class PPOMemory:
    def __init__(self, batch_size):
        self.rel_labels = []
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return np.array(self.rel_labels, dtype=object), \
               np.array(self.states, dtype=object), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, rel_labels, state, action, probs, vals, reward, done):
        self.rel_labels.append(rel_labels)
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.rel_labels = []
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=45, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, n_actions),
            nn.Softmax(dim=0)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        probs = self.actor(state)
        dist = probs.reshape(probs.shape[0])
        dist = Categorical(dist)
        return dist, probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=45,
                 chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, data, input_dims, gamma=1, alpha=0.0003, beta=0.001, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=5, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.data = data

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, beta)
        self.memory = PPOMemory(batch_size)

    def remember(self, rel_labels, state, action, probs, vals, reward, done):
        self.memory.store_memory(rel_labels, state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def compute_phi(self, critic_state, probs):
        x = np.multiply(critic_state, probs)
        phi_st = x.sum(axis=0)
        phi_st = phi_st.reshape(1, phi_st.shape[0])
        return phi_st

    def choose_action(self, observation, rel_label):
        state = T.from_numpy(observation).float().to(self.actor.device)

        # dist = self.actor(state)
        # value = self.critic(state)
        # value = value.reshape(value.shape[0])
        # value = -T.dot(value, T.tensor(rel_label).float())
        # action = dist.sample()
        # probs = T.squeeze(dist.log_prob(action)).item()
        # action = T.squeeze(action).item()
        # value = T.squeeze(value).item()
        dist, softmaxop = self.actor(state)
        phi_s = self.compute_phi(state, softmaxop.detach().numpy())

        critic_value = self.critic(phi_s)
        value = critic_value.reshape(critic_value.shape[0])
        action = dist.sample()
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        return action, probs, value

    def choose_best_action(self, observation, rel_label):
        state = T.from_numpy(observation).float().to(self.actor.device)
        dist, softmaxop = self.actor(state)
        phi_s = self.compute_phi(state, softmaxop.detach().numpy())
        critic_value = self.critic(phi_s)
        value = critic_value.reshape(critic_value.shape[0])
        action = T.tensor(np.argmax(rel_label))
        probs = T.squeeze(dist.log_prob(action)).item()
        value = T.squeeze(value).item()
        return action, probs, value

    def choose_action_test(self, observation, rel_label):
        state = T.from_numpy(observation).float().to(self.actor.device)

        # dist = self.actor(state)
        # value = self.critic(state)
        # value = value.reshape(value.shape[0])
        # value = -T.dot(value, T.tensor(rel_label).float())
        # action = dist.sample()
        # probs = T.squeeze(dist.log_prob(action)).item()
        # action = T.squeeze(action).item()
        # value = T.squeeze(value).item()
        _, softmaxop = self.actor(state)
        action = T.argmax(softmaxop)
        action = T.squeeze(action).item()
        return action

    def learn(self):

        for _ in range(self.n_epochs):

            rel_labels, state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            # print("rellabels:",rel_labels,"\nstate_arr:",state_arr,"\naction_arr:",action_arr,"\nold_prob_arr:",old_prob_arr,
            #     "\nvals_arr:",vals_arr,"\nreward_arr:",reward_arr,"\ndones_arr:",dones_arr,"\nbatches:",batches)

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                rel_labels_batch = rel_labels[batch]
                states = state_arr[batch]  # Batch size is 5
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(
                    self.actor.device)  # Problem: list of index.. [should be taken with respect to the state]
                dist = []
                critic_value = []
                for state, labels in zip(states, rel_labels_batch):
                    state_tensor = T.from_numpy(state).float().to(self.actor.device)
                    dist_n, probs_n = self.actor(state_tensor)
                    dist.append(dist_n)
                    # dist.append(self.actor(state_tensor))
                    phi_s = self.compute_phi(state_tensor.detach(), probs_n.detach().numpy())
                    critic_state = self.critic(phi_s)  # self.critic(state_tensor)
                    # critic_state = critic_state.reshape(critic_state.shape[0])
                    # value = -T.dot(critic_state, T.tensor(labels).float())
                    critic_value.append(T.squeeze(critic_state))
                # print("\ndist_new:",dist)
                # print("\ncritic vals:",critic_value)
                # exit(0)
                new_probs = []
                dist_entropy = []
                for dis, action in zip(dist, actions):
                    new_probs.append(dis.log_prob(action))
                    dist_entropy.append(dis.entropy())

                new_probs = T.stack(new_probs)
                dist_entropy = T.stack(dist_entropy)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_value = T.stack(critic_value)
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                # total_loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
