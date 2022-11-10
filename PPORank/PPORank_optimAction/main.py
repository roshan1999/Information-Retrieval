from Environment import Dataset, update_state
from Evaluation import validate_individual, calculate
from AgentPPO import Agent
import time
from progress.bar import Bar
import numpy as np
import os
import pickle
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
import random
from visualizer import visualize_train, visualize_test, visualize_rewards

torch.manual_seed(7)
np.random.seed(7)
random.seed(7)


def train(model, data, t_steps):
    model.actor.train()
    model.critic.train()
    dcg_results = {}
    ts = time.time()
    bar = Bar('Training', max=len(data.getTrain()))
    episode_rewards_list = []
    n_steps = 0
    epoch_avg_step_reward = 0
        done = False
        dcg_results[Q] = []
        state = data.getDocQuery()[Q]
        action_list = []
        episode_reward = 0
        effective_length = min(t_steps, len(state))
        learn_iters = 0
        for t in range(0, len(state)):
            observation = [data.getFeatures()[x] for x in state]
            observation = np.array(observation, dtype=float)

            # Note: Actor takes action and returns index of the state
            action, prob, val = model.choose_best_action(observation, data.getRelevance(Q, state))
            action_list.append(state[action])

            # Update to next state and get reward
            state_, reward = update_state(t, Q, state[action], state, data.getTruth())

            n_steps += 1
            episode_reward += reward
            
            if len(state_) == 0:
                done = True

            model.remember(data.getRelevance(Q, state), observation, action, prob, val, reward, done)

            if n_steps % t_steps == 0:
                model.learn()
                learn_iters += 1

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
    model.actor.eval()
    model.critic.eval()
    score = 0
    dcg_results = {}
    ts = time.time()
    bar = Bar('Initial Training', max=len(data.getTrain()))
    for Q in data.getTrain():
        dcg_results[Q] = []
        state = data.getDocQuery()[Q]
        action_list = []
        n_steps = 0
        for t in range(0, len(state)):
            observation = [data.getFeatures()[x] for x in state]
            observation = np.array(observation, dtype=float)

            # Note: Actor takes action and returns index of the state
            action = model.choose_action_test(observation, data.getRelevance(Q, state))
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
    score = 0
    dcg_results = {}
    ts = time.time()
    bar = Bar('Testing', max=len(data.getTest()))
    for Q in data.getTest():
        dcg_results[Q] = []
        state = data.getDocQuery()[Q]
        action_list = []
        n_steps = 0
        for t in range(0, len(state)):
            observation = [data.getFeatures()[x] for x in state]
            observation = np.array(observation, dtype=float)

            # Note: Actor takes action and returns index of the state
            action = model.choose_action_test(observation, data.getRelevance(Q, state))
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


def pickle_data(data, output_file):
    output_handle = open(output_file, 'wb')
    pickle.dump(data, output_handle)
    output_handle.close()


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


if __name__ == '__main__':
    # Taking arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True,
                    help="relative or absolute directory of the directory where complete dictionary exists")
    ap.add_argument("-i", "--iterations", required=True,
                    help="number of iterations to run for")
    ap.add_argument("-nf", "--features_no", required=True,
                    help="number of features in the dataset")
    ap.add_argument("-g", "--gamma", required=False, default=1,
                    help="Gamma value; default = 1")
    ap.add_argument("-alpha", "--alpha", required=False, default=0.00001,
                    help="Learning rate of the agent")
    ap.add_argument("-beta", "--beta", required=False, default=0.00002,
                    help="Learning rate of the agent")
    ap.add_argument("-hnodes", "--hidden_layer_nodes", required=False, default=46,
                    help="Number of hidden nodes ")
    ap.add_argument("-steps", "--episode_length", required=False, default=50,
                    help="Number of steps to take in each episode")
    ap.add_argument("-batch_size", "--mini_batch_size", required=False, default=5,
                    help="size of the mini batch")
    ap.add_argument("-epochs", "--epochs", required=False, default=3,
                    help="epochs to optimize surrogate")
    ap.add_argument("-clip", "--policy_clip", required=False, default=0.1,
                    help="clipping parameter")
    ap.add_argument("-lambda", "--gae_lambda", required=False, default=0.95,
                    help="smoothing factor for gae")
    args = vars(ap.parse_args())

    # Initializing arguments
    data_dir = str(args["data"])
    num_features = int(args["features_no"])
    num_iterations = int(args["iterations"])
    gamma = float(args["gamma"])
    alpha_agent = float(args["alpha"])
    beta_agent = float(args["beta"])
    hidden_layers = int(args["hidden_layer_nodes"])
    policyclip = float(args["policy_clip"])
    gaelambda = float(args["gae_lambda"])

    # Extra arguments for PPO
    T_steps = int(args["episode_length"])
    batch_size = int(args["mini_batch_size"])
    n_epochs = int(args["epochs"])

    all_data = data_dir
    data_object = Dataset(all_data)

    agent = Agent(alpha=alpha_agent, beta=beta_agent, n_actions=1, batch_size=batch_size, n_epochs=n_epochs,
                  input_dims=[num_features],
                  data=data_object, policy_clip=policyclip, gae_lambda=gaelambda)

    print("\n--- Initial ---\n")

    train_results = []
    test_results = []

    train_results.append(init_train(agent, data_object))
    test_results.append(test(agent, data_object))

    print("\n--- Training Started ---\n")
    for i in range(0, num_iterations):
        print(f"\nIteration: {i + 1}\n")
        train_results.append(train(agent, data_object, T_steps))
        test_results.append(test(agent, data_object))

    agent.save_models()

    # Saving the results
    pickle_data(train_results,
                f"./Result/{get_name(data_dir)}/train_{num_iterations}_actor_lr_{alpha_agent}_critic_lr_{beta_agent}_g_{gamma}_hnodes_{hidden_layers}_T_{T_steps}_mbsize_{batch_size}_epochs_{n_epochs}")
    pickle_data(test_results,
                f"./Result/{get_name(data_dir)}/test_{num_iterations}_actor_lr_{alpha_agent}_critic_lr_{beta_agent}_g_{gamma}_hnodes_{hidden_layers}_T_{T_steps}_mbsize_{batch_size}_epochs_{n_epochs}")
    outfile = f"./Result/{get_name(data_dir)}/plot_train_{num_iterations}_actor_lr_{alpha_agent}_critic_lr_{beta_agent}_g_{gamma}_hnodes_{hidden_layers}_T_{T_steps}_mbsize_{batch_size}_epochs_{n_epochs}"
    visualize_train(train_results, outfile)
    outfile = f"./Result/{get_name(data_dir)}/plot_test_{num_iterations}_actor_lr_{alpha_agent}_critic_lr_{beta_agent}_g_{gamma}_hnodes_{hidden_layers}_T_{T_steps}_mbsize_{batch_size}_epochs_{n_epochs}"
    visualize_test(test_results, outfile)
    outfile = f"./Result/{get_name(data_dir)}/plot_rewards_{num_iterations}_actor_lr_{alpha_agent}_critic_lr_{beta_agent}_g_{gamma}_hnodes_{hidden_layers}_T_{T_steps}_mbsize_{batch_size}_epochs_{n_epochs}"
    visualize_rewards(train_results, outfile)
