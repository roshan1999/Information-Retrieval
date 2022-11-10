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
from visualizer import visualize

torch.manual_seed(4)
np.random.seed(4)
random.seed(4)


def train(model, data, t_steps, c1, c2):
    model.actor.train()
    model.critic.train()
    dcg_results = {}
    ts = time.time()
    bar = Bar('Training', max=len(data.getTrain()))
    episode_rewards_list = []
    epoch_avg_step_reward = 0
    for Q in data.getTrain():
        done = False
        dcg_results[Q] = []
        state = data.getDocQuery()[Q]
        action_list = []
        episode_reward = 0
        learn_iters = 0
        effective_length = min(t_steps, len(state)-1)
        for t in range(0, min(t_steps, len(state)-1)):
            observation = [data.getFeatures()[x] for x in state]
            observation = np.array(observation, dtype=float)

            # Note: Actor takes action and returns index of the state
            action, prob, val = model.choose_action(observation, data.getRelevance(Q, state))
            action_list.append(state[action])

            # Update to next state and get reward
            state_, reward = update_state(t, Q, state[action], state, data.getTruth())

            episode_reward += reward

            if t == min(t_steps-1, len(state)-2):
                done = True

            model.remember(data.getRelevance(Q, state), observation, action, prob, val, reward, done)


            # Update state
            state = state_

        model.learn(c1, c2)

        epoch_avg_step_reward += episode_reward / effective_length
        episode_rewards_list.append(episode_reward / effective_length)

        # Update Query DCG results:
        dcg_results[Q] = validate_individual(data.getTruth()[Q], data.getIDCG()[Q], action_list)
        dcg_results[Q] = np.round(dcg_results[Q], 4)

        bar.next()

    bar.finish()
    print(f'\n Average step reward: {round(epoch_avg_step_reward / len(data.getTrain()), 4)}, time: {round(time.time() - ts)}')
    final_result = calculate(dcg_results)
    print(f"NDCG@1: {final_result[0]}\t"
          f"NDCG@3: {final_result[2]}\tNDCG@5: {final_result[4]}\tNDCG@10: {final_result[9]}")

    return final_result, episode_rewards_list

def train_init(model, data):
    model.actor.eval()
    model.critic.eval()
    score = 0
    dcg_results = {}
    ts = time.time()
    bar = Bar('Train', max=len(data.getTrain()))
    for Q in data.getTrain():
        dcg_results[Q] = []
        state = data.getDocQuery()[Q]
        action_list = []
        for t in range(0, len(state)):
            observation = [data.getFeatures()[x] for x in state]
            observation = np.array(observation, dtype=float)

            # Note: Actor takes action and returns index of the state
            action = model.choose_action_test(observation, data.getRelevance(Q, state))
            action_list.append(state[action])

            # Update to next state and get reward
            state_, _ = update_state(t, Q, state[action], state, data.getTruth())

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
            action = model.choose_action_test(observation, data.getRelevance(Q, state))
            action_list.append(state[action])

            # Update to next state and get reward
            state_, _ = update_state(t, Q, state[action], state, data.getTruth())

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


def vali(model, data):
    model.actor.eval()
    model.critic.eval()
    score = 0
    dcg_results = {}
    ts = time.time()
    # bar = Bar('Validating', max=len(data.getVali()))
    for Q in data.getVali():
        dcg_results[Q] = []
        state = data.getDocQuery()[Q]
        action_list = []
        for t in range(0, len(state)):
            observation = [data.getFeatures()[x] for x in state]
            observation = np.array(observation, dtype=float)

            # Note: Actor takes action and returns index of the state
            action = model.choose_action_test(observation, data.getRelevance(Q, state))
            action_list.append(state[action])

            # Update to next state and get reward
            state_, _ = update_state(t, Q, state[action], state, data.getTruth())

            # Update state
            state = state_

        # Update Query DCG results:
        dcg_results[Q] = validate_individual(data.getTruth()[Q], data.getIDCG()[Q], action_list)
        dcg_results[Q] = np.round(dcg_results[Q], 4)

        # bar.next()

    # bar.finish()
    # print(f'\n TotalReward: {round(np.array(score_history).mean(), 4)}, time: {round(time.time() - ts)}')
    final_result = calculate(dcg_results)
    # print(f"NDCG@1: {final_result[0]}\t"
          # f"NDCG@3: {final_result[2]}\tNDCG@5: {final_result[4]}\tNDCG@10: {final_result[9]}")

    return final_result


def pickle_data(data, output_file):
    output_handle = open(output_file, 'wb')
    pickle.dump(data, output_handle)
    output_handle.close()

def get_name(datadir):
    lst=datadir.split('/')
    ds=""
    for i in lst:
        if('ohsumed' in i.lower()):
            ds='OHSUMED'
        elif('mq2008' in i.lower()):
            ds='MQ2008'
        elif('mq2007' in i.lower()):
            ds='MQ2007'
        elif('mslr-web10k' in i.lower()):
            ds='MSLR-WEB10K'
    if(len(ds)==0):
        print("Wrong Dataset,Please check path")
        exit()
    else:
        return ds

if __name__ == '__main__':
    # Taking arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True,
                    help="relative or absolute directory of the directory where complete dictionary exists")
    ap.add_argument("-f", "--fold", required=True,
                    help="Fold")
    ap.add_argument("-i", "--iterations", required=True,
                    help="number of iterations to run for")
    ap.add_argument("-nf", "--features_no", required=True,
                    help="number of features in the dataset")
    ap.add_argument("-g", "--gamma", required=False, default=1,
                    help="Gamma value; default = 1")
    ap.add_argument("-alpha", "--alpha", required=False, default=0.0001,
                    help="Learning rate of the actor ")
    ap.add_argument("-beta", "--beta", required=False, default=0.0002,
                    help="Learning rate of the critic ")
    ap.add_argument("-hnodes", "--hidden_layer_nodes", required=False, default=45,
                    help="Number of hidden nodes ")
    ap.add_argument("-steps", "--episode_length", required=False, default=20,
                    help="Number of steps to take in each episode")
    ap.add_argument("-batch_size", "--mini_batch_size", required=False, default=5,
                    help="size of the mini batch")
    ap.add_argument("-epochs", "--epochs", required=False, default=1,
                    help="epochs to optimize surrogate")
    ap.add_argument("-clip", "--policy_clip", required=False, default=0.1,
                    help="clipping parameter")
    ap.add_argument("-lambda", "--gae_lambda", required=False, default=1.0,
                    help="smoothing factor for gae")
    ap.add_argument("-c1", "--VF_coef", required=False, default=0.5,
                    help="VF coef")
    ap.add_argument("-c2", "--S_coef", required=False, default=0.01,
                    help="Entropy coef")
    args = vars(ap.parse_args())

    # Initializing arguments
    data_dir = str(args["data"])
    fold = int(args["fold"])
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
    c1 = float(args["VF_coef"])
    c2 = float(args["S_coef"])

    all_data = data_dir + f'/Fold{fold}/'
    data_object = Dataset(all_data, num_features)

    agent = Agent(alpha=alpha_agent, beta=beta_agent, n_actions=1, batch_size=batch_size, n_epochs=n_epochs,
                  input_dims=[num_features],
                  data=data_object, policy_clip=policyclip, gae_lambda=gaelambda, nodes=hidden_layers)

    
    train_results = []
    vali_results = []
    test_results = []

    print("\n--- Training Started ---\n")
    for i in range(0, num_iterations):
        print(f"\nIteration: {i + 1}\n")

        train_results.append(train(agent, data_object, T_steps, c1, c2))
        print()
        test_results.append(test(agent, data_object))
        print()
        vali_results.append(vali(agent, data_object))

    agent.save_models()    

    filename = f"./Result/{get_name(data_dir)}/train_fold_{all_data[-2]}_iter_{num_iterations}_clip_{policyclip}_c1_{c1}_c2_{c2}_lambda_{gaelambda}_actor_lr_{alpha_agent}_critic_lr_{beta_agent}_g_{gamma}_hnodes_{hidden_layers}_T_{T_steps}_mbsize_{batch_size}_epochs_{n_epochs}"

    pickle_data(train_results, filename)
    
    pickle_data(vali_results, f"./Result/{get_name(data_dir)}/vali_fold_{all_data[-2]}_iter_{num_iterations}_clip_{policyclip}_c1_{c1}_c2_{c2}_lambda_{gaelambda}__actor_lr_{alpha_agent}_critic_lr_{beta_agent}_g_{gamma}_hnodes_{hidden_layers}_T_{T_steps}_mbsize_{batch_size}_epochs_{n_epochs}")
    
    pickle_data(test_results,
                f"./Result/{get_name(data_dir)}/test_fold_{all_data[-2]}_iter_{num_iterations}_clip_{policyclip}_c1_{c1}_c2_{c2}_lambda_{gaelambda}_actor_lr_{alpha_agent}_critic_lr_{beta_agent}_g_{gamma}_hnodes_{hidden_layers}_T_{T_steps}_mbsize_{batch_size}_epochs_{n_epochs}")
    
    outfile = f"./Result/{get_name(data_dir)}/plot_train_fold_{all_data[-2]}_iter_{num_iterations}_clip_{policyclip}_c1_{c1}_c2_{c2}_lambda_{gaelambda}_actor_lr_{alpha_agent}_critic_lr_{beta_agent}_g_{gamma}_hnodes_{hidden_layers}_T_{T_steps}_mbsize_{batch_size}_epochs_{n_epochs}"

    visualize(train_results, outfile)