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

torch.manual_seed(4)
np.random.seed(4)
random.seed(4)


def test(model, data):
    model.actor.eval()
    model.critic.eval()
    score = 0
    dcg_results = {}
    ts = time.time()
    bar = Bar('Testing', max=len(data.getTest()))
    score_history = []
    for Q in data.getTest():
        dcg_results[Q] = []
        state = data.getDocQuery()[Q]
        action_list = []
        score = 0
        n_steps = 0
        for t in range(0, len(state)):
            observation = [data.getFeatures()[x] for x in state]
            observation = np.array(observation, dtype=float)

            # Note: Actor takes action and returns index of the state
            action, _, _, _ = model.choose_action(observation, data.getRelevance(Q, state))
            action_list.append(state[action])

            # Update to next state and get reward
            state_, reward = update_state(t, Q, state[action], state, data.getTruth())

            n_steps += 1
            score += reward

            # Update state
            state = state_

        score_history.append(score)

        # Update Query DCG results:
        dcg_results[Q] = validate_individual(data.getTruth()[Q], data.getIDCG()[Q], action_list)
        dcg_results[Q] = np.round(dcg_results[Q], 4)

        bar.next()

    bar.finish()
    print(f'\n TotalReward: {round(np.array(score_history).mean(), 4)}, time: {round(time.time() - ts)}')
    final_result = calculate(dcg_results)
    print(f"NDCG@1: {final_result[0]}\t"
          f"NDCG@3: {final_result[2]}\tNDCG@5: {final_result[4]}\tNDCG@10: {final_result[9]}")

    return final_result, np.array(score_history).mean()



def pickle_data(data, output_file):
    output_handle = open(output_file, 'wb')
    pickle.dump(data, output_handle)
    output_handle.close()

def view_save_plot(train_results, outfile):
    lst_returns = [x[1] for x in train_results]
    smooth_returns = pd.Series.rolling(pd.Series(lst_returns), 10).mean()
    plt.title("Total Returns")
    plt.plot(lst_returns, markersize='5', color='blue', label='Total Returns')
    plt.plot(smooth_returns, linewidth=3, markersize='5', color='green', label='Smooth Total Returns')
    plt.legend()
    plt.savefig(outfile)
    plt.show()


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
    ap.add_argument("-alpha", "--alpha", required=False, default=0.0005,
                    help="Learning rate of the agent; Defaults to 0.008")
    ap.add_argument("-hnodes", "--hidden_layer_nodes", required=False, default=35,
                    help="Number of hidden nodes ")
    ap.add_argument("-steps", "--episode_length", required=False, default=20,
                    help="Number of steps to take in each episode")
    ap.add_argument("-batch_size", "--mini_batch_size", required=False, default=5,
                    help="size of the mini batch")
    ap.add_argument("-epochs", "--epochs", required=False, default=4,
                    help="epochs to optimize surrogate")
    ap.add_argument("-clip", "--policy_clip", required=False, default=0.2,
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
    hidden_layers = int(args["hidden_layer_nodes"])
    policyclip = float(args["policy_clip"])
    gaelambda = float(args["gae_lambda"])

    # Extra arguments for PPO
    T_steps = int(args["episode_length"])
    batch_size = int(args["mini_batch_size"])
    n_epochs = int(args["epochs"])

    all_data = data_dir
    data_object = Dataset(all_data)

    agent = Agent(alpha=alpha_agent, n_actions=1, batch_size=batch_size, n_epochs=n_epochs,
                  input_dims=[num_features],
                  data=data_object, policy_clip=policyclip, gae_lambda=gaelambda)

    print("\n--- Training Started ---\n")
    """
    # agent.actor.load_state_dict(torch.load(open('model_actor', 'rb')))
    # agent.critic.load_state_dict(torch.load(open('model_critic', 'rb')))
    train_results, all_rewards = train(agent, num_epochs, data_object)
    # save_f_actor = open('model_actor', 'wb')
    # save_f_critic = open('model_critic', 'wb')
    # torch.save(agent.actor.state_dict(), save_f_actor)
    # torch.save(agent.critic.state_dict(), save_f_critic)
    
    print("\n--- Training Completed ---\n")
    print("\n--- Validation Started ---\n")
    vali_results = vali(agent, data_object)
    print("\n--- Validation Completed ---\n")
    print("\n--- Testing Started ---\n")
    test_results = test(agent, data_object)
    print("\n--- Testing Completed ---\n")
    """
    test_results = []
    # for i in range(0, num_iterations):
    # print(f"\nIteration: {i + 1}\n")
    agent.load_models()
    test_results.append(test(agent, data_object))
    