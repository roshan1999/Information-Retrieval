from Environment import Dataset, update_state
from Evaluation import validate_individual, calculate
from Agent import Agent
import time
from progress.bar import Bar
import numpy as np
import os
import torch
import pickle
import argparse
import random
from visualizer import visualize_test, visualize_train, visualize_rewards

np.random.seed(4)
random.seed(4)
torch.manual_seed(4)


# Training the model
def train(model, data, episode_length):

    # Setting pytorch to training mode
    model.actor.train()
    model.critic.train()

    # Buffer for rewards to perform average later
    episode_rewards_list = []
    epoch_avg_step_reward = 0

    dcg_results = {}
    ts = time.time()
    bar = Bar('Training', max=len(data.getTrain()))

    # For each query in the training set
    for Q in data.getTrain():
        dcg_results[Q] = []

        # Initialize state with Total number of documents
        state = data.getDocQuery()[Q]
        action_list = []
        episode_reward = 0
        effective_length = min(episode_length, len(state))

        for t in range(0, effective_length):

            # Converting the current state into numpy array to make into tensors later
            observation = [data.getFeatures()[x] for x in state]
            observation = np.array(observation, dtype=float)

            # Actor chooses an action (a document at t position)
            action = model.choose_action(observation)
            
            # all actions stored in buffer for calculation of DCG scores later
            action_list.append(state[action])

            # Get the next state and the reward based on the action
            state_, reward = update_state(t, Q, state[action], state, data.getTruth())

            episode_reward += reward
            observation_ = [data.getFeatures()[x] for x in state_]
            observation_ = np.array(observation_, dtype=float)

            # Update agent parameters
            model.update(observation, reward, observation_)

            # Update state
            state = state_

        epoch_avg_step_reward += episode_reward / effective_length
        episode_rewards_list.append(episode_reward / effective_length)

        # Update Query DCG results:
        dcg_results[Q] = validate_individual(data.getTruth()[Q], data.getIDCG()[Q], action_list)
        dcg_results[Q] = np.round(dcg_results[Q], 4)

        bar.next()

    bar.finish()
    print(
    f'\n Average step reward: {round(epoch_avg_step_reward / len(data.getTrain()), 4)}, time: {round(time.time() - ts)}')
    final_result = calculate(dcg_results)
    print(f"NDCG@1: {final_result[0]}\t"
          f"NDCG@3: {final_result[2]}\tNDCG@5: {final_result[4]}\tNDCG@10: {final_result[9]}")

    return final_result, episode_rewards_list


def init_train(model, data):

    # Setting pytorch to evaluation mode (stops updating the gradients)
    model.actor.eval()
    model.critic.eval()

    # Buffer for rewards to perform average later
    ep_reward = 0
    episode_rewards_list = []
    epoch_avg_step_reward = 0

    dcg_results = {}
    ts = time.time()
    bar = Bar('Initial Training', max=len(data.getTrain()))

    # For each query in the training set
    for Q in data.getTrain():
        dcg_results[Q] = []
        
        # Initialize state with Total number of documents
        state = data.getDocQuery()[Q]
        action_list = []
        episode_reward = 0
        effective_length = len(state)

        for t in range(0, len(state)):
            
            # Converting the current state into numpy array to make into tensors later
            observation = [data.getFeatures()[x] for x in state]
            observation = np.array(observation, dtype=float)

            # Actor chooses an action (a document at t position)
            action = model.choose_action_test(observation)

            # all actions stored in buffer for calculation of DCG scores later
            action_list.append(state[action])

            # Get the next state and the reward
            state_, reward = update_state(t, Q, state[action], state, data.getTruth())

            episode_reward += reward

            # Update state
            state = state_

        # Update Query DCG results:
        dcg_results[Q] = validate_individual(data.getTruth()[Q], data.getIDCG()[Q], action_list)
        dcg_results[Q] = np.round(dcg_results[Q], 4)

        # Average the rewards with by the number of documents
        epoch_avg_step_reward += episode_reward / effective_length
        episode_rewards_list.append(episode_reward / effective_length)

        bar.next()

    bar.finish()
    print(
    f'\n Average step reward: {round(epoch_avg_step_reward / len(data.getTrain()), 4)}, time: {round(time.time() - ts)}')
    final_result = calculate(dcg_results)
    print(f"NDCG@1: {final_result[0]}\t"
          f"NDCG@3: {final_result[2]}\tNDCG@5: {final_result[4]}\tNDCG@10: {final_result[9]}")

    return final_result, episode_rewards_list


def test(model, data):
    
    # Setting pytorch to evaluation mode (stops updating the gradients)
    model.actor.eval()
    model.critic.eval()
    ep_reward = 0
    dcg_results = {}
    ts = time.time()
    bar = Bar('Testing', max=len(data.getTest()))

    for Q in data.getTest():
        dcg_results[Q] = []
        state = data.getDocQuery()[Q]
        action_list = []

        Q_reward = 0
        for t in range(0, len(state)):
            observation = [data.getFeatures()[x] for x in state]
            observation = np.array(observation, dtype=float)

            # Note: Actor takes action and returns index of the state
            action = model.choose_action_test(observation)
            action_list.append(state[action])

            # Update to next state and get reward
            state_, reward = update_state(t, Q, state[action], state, data.getTruth())

            # Update state
            state = state_

        # Update Query DCG results:
        dcg_results[Q] = validate_individual(data.getTruth()[Q], data.getIDCG()[Q], action_list)
        dcg_results[Q] = np.round(dcg_results[Q], 4)
        bar.next()

    bar.finish()
    final_result = calculate(dcg_results)
    print(f"NDCG@1: {final_result[0]}\t"
          f"NDCG@3: {final_result[2]}\tNDCG@5: {final_result[4]}\tNDCG@10: {final_result[9]}")

    return final_result

def get_name(datadir):
    lst = datadir.split('/')
    ds = ""
    for i in lst:
        if 'ohsumed' in i.lower():
            ds = 'OHSUMED'
        elif 'mq2008' in i.lower():
            ds = 'MQ2008'
        elif 'mq2007' in i.lower():
            ds = 'MQ2007'
        elif 'mslr-web10k' in i.lower():
            ds = 'MSLR-WEB10K'
    if len(ds) == 0:
        print("Wrong Dataset,Please check path")
        exit()
    else:
        return ds

def pickle_data(data, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)

    output_handle = open(output_file, 'wb')
    pickle.dump(data, output_handle)
    output_handle.close()


if __name__ == '__main__':

    # Taking arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True,
                    help="relative or absolute directory of the directory where fold exists")
    ap.add_argument("-e", "--epoch", required=True,
                    help="number of epochs to run for")
    ap.add_argument("-nf", "--features_no", required=True,
                    help="number of features in the dataset")
    ap.add_argument("-g", "--gamma", required=False, default=1,
                    help="Gamma value; default = 1")
    ap.add_argument("-alpha", "--alpha", required=False, default=1e-05,
                    help="Learning rate of the actor")
    ap.add_argument("-beta", "--beta", required=False, default=2e-05,
                    help="Learning rate of the critic")
    ap.add_argument("-length", "--episode_length", required=False, default=20,
                    help="Episode length")
    args = vars(ap.parse_args())

    # Initializing arguments
    data_dir = str(args["data"])
    num_features = int(args["features_no"])
    num_epochs = int(args["epoch"])
    gamma = float(args["gamma"])
    alpha = float(args["alpha"])
    beta = float(args["beta"])
    episode_length = int(args['episode_length'])

    # Agent object initialization
    agent = Agent(actor_lr=alpha, critic_lr=beta, input_dims=num_features,
                  gamma=gamma)

    # Dataset object initialization
    dataset = f"{data_dir}"
    data_object = Dataset(dataset)

    print("\n--- Training Started ---\n")

    train_results = []
    test_results = []

    # Initial results on the current weights
    train_results.append(init_train(agent, data_object))
    test_results.append(test(agent, data_object))

    for i in range(0, num_epochs):
        print(f"\nEpoch: {i+1}\n")
        train_results.append(train(agent, data_object, episode_length))
        test_results.append(test(agent, data_object))


    # Saving all results
    pickle_data(train_results, f"./Result/{get_name(data_dir)}/_train_epochs{num_epochs}_a,c{alpha},{beta}_g{gamma}_length{episode_length}")

    pickle_data(test_results, f"./Result/{get_name(data_dir)}/_test_epochs{num_epochs}_a,c{alpha},{beta}_g{gamma}_length{episode_length}")

    outfile = f"./Result/{get_name(data_dir)}/_train_epochs{num_epochs}_a,c{alpha},{beta}_g{gamma}_length{episode_length}"
    visualize_train(train_results, outfile)

    outfile = f"./Result/{get_name(data_dir)}/_test_epochs{num_epochs}_a,c{alpha},{beta}_g{gamma}_length{episode_length}"
    visualize_test(test_results, outfile)
    
    outfile = f"./Result/{get_name(data_dir)}/_rewards_epochs{num_epochs}_a,c{alpha},{beta}_g{gamma}_length{episode_length}"
    visualize_rewards(train_results, outfile)