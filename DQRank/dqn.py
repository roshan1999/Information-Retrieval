import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    """Deep Q Network"""
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, 
            n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.net= nn.Sequential(nn.Linear(*self.input_dims, self.fc1_dims),
                        nn.ReLU(),
                        #nn.Linear(self.fc1_dims, self.fc2_dims),
                        #nn.ReLU(),
                        nn.Linear(self.fc1_dims, self.n_actions))

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        actions=self.net(state)
        return actions


class Agent():
    """Define the RL Agent"""
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
            max_mem_size=100000,hnodes=256, eps_end=0.05, eps_dec=5e-4,replace_target=100):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = replace_target
        self.hnodes=hnodes

        #Initialize the Evaluation and target networks
        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                    fc1_dims=hnodes, fc2_dims=hnodes)
        self.Q_next = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                    fc1_dims=hnodes, fc2_dims=hnodes)

        #Initialize the buffers
        self.state_memory = [0]*self.mem_size
        self.new_state_memory = [0]*self.mem_size
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        """Store the current transition in the buffer"""
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        """Perform episilon greedy selection of action to take(policy in dqn)."""
        if np.random.random() > self.epsilon:
            state = observation.float().to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(len(observation))

        return action

    def choose_action_test(self, observation):
        """Pick greedy action wrt to Q network for testing"""
        with torch.no_grad():
            state = observation.float().to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        
        return action

    def qvals_minibatch(self,state_batch,action_batch):
        """Returns the qvals of the actions taken in that batch"""
        qvals=[]
        for s,a in zip(state_batch,action_batch):
            qvalues = self.Q_eval.forward(s)
            qvals.append(qvalues[a])

        return T.squeeze(T.stack(qvals))

    def qvals_minibatch_target(self,state_batch):
        """Returns the qvals for every action in the given states"""
        qvals=[]
        for s in state_batch:
            if(len(s)>0):
                qvalues = self.Q_next.forward(s)
                a = T.argmax(qvalues).item()
                qvals.append(qvalues[a])
            else:
                qvals.append(T.tensor([0.0],requires_grad=True))
 
        return T.squeeze(T.stack(qvals))


    def learn(self):
        """Samples a minibatch and updates the q-networks"""

        #If the transitions in the buffer less than batch size then return
        if self.mem_cntr < self.batch_size:
            return

        #Clear any gradients
        self.Q_eval.optimizer.zero_grad()
        
        #Find no. of transitions in the buffer
        max_mem = min(self.mem_cntr, self.mem_size)

        #Pick a random mini batch
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        #print("max_mem:",max_mem,"mem_cntr:",self.mem_cntr)

        #Get the current mini-batch
        state_batch=[]
        new_state_batch=[]
        for index in batch:
            state_batch.append(self.state_memory[index])
            new_state_batch.append(self.new_state_memory[index])
        
        state_batch=state_batch
        new_state_batch = new_state_batch
        action_batch = self.action_memory[batch]
        
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        
        #Get the qvals 
        q_eval=self.qvals_minibatch(state_batch,action_batch)
        q_next = self.qvals_minibatch_target(new_state_batch)
        q_next[terminal_batch] = 0.0

        #Compute the q-learning target
        q_target = reward_batch + self.gamma*q_next

        #Compute the loss
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

        #Update parameters by SGD and backprop
        loss.backward()
        self.Q_eval.optimizer.step()

        #Increase the update counter
        self.iter_cntr += 1

        #Decay your epsilon 
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                       else self.eps_min

        #Update the target network
        if self.iter_cntr % self.replace_target == 0:
          self.Q_next.load_state_dict(self.Q_eval.state_dict())
