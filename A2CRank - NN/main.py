from Environment import Dataset, update_state
from Evaluation import validate_individual, calculate
from Agent import Agent
import time
from progress.bar import Bar
import numpy as np
import os
import pickle
import argparse
import torch
import random
from visualizer import visualize_test, visualize_train, visualize_rewards

# Seeding
torch.manual_seed(543)
np.random.seed(543)
random.seed(543)


# Training the model
def train(model, data):
    # Setting pytorch to training mode
    model.actor.train()
    model.critic.train()

    # Reward for average later
    ep_reward = 0
    episode_rewards_list = []
    epoch_avg_step_reward = 0

    # Dictionary for storing dcg values
    dcg_results = {}
    ts = time.time()
    bar = Bar('Training', max=len(data.getTrain()))

    # Start training
    for Q in data.getTrain():
        dcg_results[Q] = []
        # Total number of documents = number of transitions
        state = data.getDocQuery()[Q]
        # Storing for dcg calculation later
        action_list = []
        episode_reward = 0
        # Done indicates that the entire length has been traversed
        done = False
        effective_length = len(state)

        # Rank each document, update in the end
        for t in range(0, effective_length):

            # Converting documnets to respective features 
            observation = [data.getFeatures()[x] for x in state]
            observation = np.array(observation, dtype=float)

            # Note: Actor takes action and returns index of the state
            action = model.choose_action(observation)
            action_list.append(state[action])

            # Update to next state and get reward
            state_, reward = update_state(t, Q, state[action], state, data.getTruth())
            episode_reward += reward

            # Getting features for the next state
            observation_ = [data.getFeatures()[x] for x in state_]
            observation_ = np.array(observation_, dtype=float)

            # Checking if last state completed
            if len(observation_) == 0:
                done = True

            # Store all rewards for updation later
            model.store_values(reward, observation)

            # Update state
            state = state_

        # Update at end of episode 
        model.update()

        # Take average of rewards first with episode then with total queries
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

    # return results_epoch, rewards_history
    return final_result, episode_rewards_list

def init_train(model, data):
    # Setting model to evaluation mode
    model.actor.eval()
    model.critic.eval()
    dcg_results = {}
    ts = time.time()
    bar = Bar('Initial Training', max=len(data.getTrain()))

    # Performign initial training for each Query
    for Q in data.getTrain():
        dcg_results[Q] = []
        state = data.getDocQuery()[Q]
        action_list = []

        # Rank each document, no update in the end
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

    return final_result, [0]

def test(model, data):
    model.actor.eval()
    model.critic.eval()
    dcg_results = {}
    ts = time.time()
    bar = Bar('Testing', max=len(data.getTest()))

    for Q in data.getTest():
        dcg_results[Q] = []
        state = data.getDocQuery()[Q]
        action_list = []

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

# Saves to the corresponding folder by taking data directory as input
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
    ap.add_argument("-alpha", "--alpha", required=False, default=0.0001,
                    help="Learning rate of the actor")
    ap.add_argument("-beta", "--beta", required=False, default=0.0002,
                    help="Learning rate of the critic")
    ap.add_argument("-layer", "--hidden_layer", required=False, default=46,
                    help="Number of hidden nodes ")
    args = vars(ap.parse_args())

    # Initializing arguments
    data_dir = str(args["data"])
    num_features = int(args["features_no"])
    num_epochs = int(args["epoch"])
    gamma = float(args["gamma"])
    actor_lr = float(args["alpha"])
    critic_lr = float(args["beta"])
    hnodes = int(args["hidden_layer"])
    agent = Agent(actor_lr, critic_lr, input_dims=[num_features],
                  gamma=gamma, n_actions=1, layer1_size=hnodes, layer2_size=hnodes)
    seed = 543

    # Create the dataset object
    all_data = data_dir
    data_object = Dataset(all_data)

    # Store all results to file later from list
    train_results = []
    test_results = []

    # Initial values of training and testing
    train_results.append(init_train(agent, data_object))
    test_results.append(test(agent, data_object))

    # Starting training and testing for each epoch
    for i in range(0, num_epochs):
        print(f"\nEpoch: {i+1}\n")
        train_results.append(train(agent, data_object))
        test_results.append(test(agent, data_object))

    # Storing all results
    pickle_data(train_results, f"./Result/{get_name(data_dir)}/_train_epochs{num_epochs}_a,c{actor_lr},{critic_lr}_g{gamma}_node{hnodes}_seed_{seed}")

    pickle_data(test_results, f"./Result/{get_name(data_dir)}/_test_epochs{num_epochs}_a,c{actor_lr},{critic_lr}_g{gamma}_node{hnodes}_seed_{seed}")

    outfile = f"./Result/{get_name(data_dir)}/_train_epochs{num_epochs}_a,c{actor_lr},{critic_lr}_g{gamma}_node{hnodes}_seed_{seed}"
    visualize_train(train_results, outfile)

    outfile = f"./Result/{get_name(data_dir)}/_test_epochs{num_epochs}_a,c{actor_lr},{critic_lr}_g{gamma}_node{hnodes}_seed_{seed}"
    visualize_test(test_results, outfile)
    
    outfile = f"./Result/{get_name(data_dir)}/_rewards_epochs{num_epochs}_a,c{actor_lr},{critic_lr}_g{gamma}_node{hnodes}_seed_{seed}"
    visualize_rewards(train_results, outfile)