import os
import glob

import torch
import numpy as np

from PPO import PPO
from Environment import *

class PPORank:
    def __init__(self):
        return

    def rank(self, curr_docLst, ppo_agent, data, save_model_freq=1, update_timestep=1):
        
        #Monitor timestep to update and save models
        self.time_step = 0
        self.update_timestep = update_timestep
        self.save_model_freq = save_model_freq

        #Docids in the current state
        state = curr_docLst.copy()

        action_list=[]
        episode_list = []
        effective_length=len(state)

        for t in range(0, effective_length):

            #Feature matrix of the current state
            observation = [data.getFeatures()[x] for x in state]
            observation = np.array(observation, dtype=float)

            # select action with policy
            action = ppo_agent.select_action(observation)
            episode_list.append(([t, state], state[action]))
            action_list.append(state[action])
            
            #Update state based on action and get reward
            state_ = update_state(t, state[action], state)
            
            #Check if terminal state
            if(len(state_)==0):
                done=True
            else:
                done=False

            #Check if the epsiode is truncated here
            if(t==len(state)-1):
                is_ep_trunc=True
            else:
                is_ep_trunc=False

            #saving reward and is_terminals and trunc history
            # ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            ppo_agent.buffer.is_ep_trunc.append(is_ep_trunc)

            self.time_step +=1

            # # update PPO agent
            # if time_step % update_timestep == 0:
            #     ppo_agent.update()

            # # save model weights
            # if time_step % save_model_freq == 0:
            #     ppo_agent.save(checkpoint_path)

            #If episode over then break    
            if done:
                break

            #Update the state
            state=state_

        return episode_list, action_list


    def update(self, ppo_agent, userFeedbacks, doc_M_rs, query_doc_N_dict,N):
        # For each query optimize the weights
        
        #To monitor the number of querys involved in the update
        counter=0
        for q in userFeedbacks:

            # For each state calculate return and delta_w
            self.gen_store_rewards(query_doc_N_dict[q[1]], doc_M_rs, ppo_agent)
            
            # update PPO agent
            if self.time_step % self.update_timestep == 0:
                ppo_agent.update(counter=counter,topK=N)

            # save model weights
            if self.time_step % self.save_model_freq == 0:
                spath=f"./Results/_{self.time_step}"
                checkpoint_path = spath + "_model"
                #ppo_agent.save(checkpoint_path)
            #To monitor the query number
            counter+=1

        #Clear the ppo agent buffer
        ppo_agent.buffer.clear()

    # Generating all rewards for the list of documents on screen
    def gen_store_rewards(self, docLst, doc_M_rs, ppo_agent):
        for index, doc in enumerate(docLst):
            ppo_agent.buffer.rewards.append(single_reward(doc_M_rs[doc], index))

# Generate reward based on the relevance score and the position of the documents
def single_reward(score, index):
    if index == 0:
        return (2 ** score) - 1
    else:
        return ((2 ** score) - 1) / float(math.log((index + 1), 2))

