import gym
from Agent import *
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import argparse

np.random.seed(0)


if __name__ == '__main__':

    # Arguments parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--episodes", required=False, default=2500,
                    help="number of episodes to train for")
    ap.add_argument("-g", "--gamma", required=False, default=0.99,
                    help="Gamma value; default = 0.99")
    ap.add_argument("-alpha", "--actor_lr", required=False, default=1e-4,
                    help="Learning rate of the actor; Defaults to 0.0001")
    ap.add_argument("-beta", "--critic_lr", required=False, default=5e-4,
                    help="Learning rate of the agent; Defaults to 0.0005")
    ap.add_argument("-layer1", "--hidden_layer1_size", required=False, default=32,
                    help="Number of nodes in layer 1")
    ap.add_argument("-layer2", "--hidden_layer2_size", required=False, default=32,
                    help="Number of nodes in layer 2")
    ap.add_argument("-interval", "--interval", required=False, default=20,
                    help="Interval to display the total rewards")
    args = vars(ap.parse_args())

    # All the hyper-parameters
    num_episodes = int(args["episodes"])
    gamma = float(args["gamma"])
    actor_lr = float(args["actor_lr"])
    critic_lr = float(args["critic_lr"])
    lay1_size = int(args["hidden_layer1_size"])
    lay2_size = int(args["hidden_layer2_size"])
    interval = int(args["interval"])

    # Displaying the hyper-parameters
    print("************************************")
    print(f"\nHyperParameters used: \n\n"
          f"number of episodes = {num_episodes} \ngamma = {gamma} "
          f"\nactor learning rate = {actor_lr} \ncritic learning rate = {critic_lr} "
          f"\nsize of 1st hidden layer = {lay1_size} "
          f"\nsize of 2nd hidden layer = {lay2_size}\n"
          f"interval for display = {interval}\n")
    print("\n*************************************")

    # Constructing the environment
    env = gym.make('CartPole-v0')

    # Taking a seed
    env.seed(0)

    # Taking the observation space and the action space for the neural network
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    # Recording all total rewards for each episode to plot graph at end
    episodic_reward_history = []

    # Initializing model
    model = Agent(actor_lr, critic_lr, gamma, state_space, action_space, lay1_size, lay2_size)

    # Running reward
    running_reward = 0

    for i in range(num_episodes):

        # Resetting to initial state
        curr_state = env.reset()

        # done flag indicates if the terminal state is reached
        done = False

        # Calculating the reward for current episode
        episode_reward = 0

        while not done:
            # Select an action
            action = model.select_action(curr_state)

            # Get the next state and the reward
            next_state, reward, done, _ = env.step(action)

            # Performing TD(0) update
            model.update(curr_state, next_state, reward, done)

            # Transition to the next state
            curr_state = next_state

            # Using the total rewards to allow for plotting
            episode_reward += reward

        # Running reward
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Appending the total reward to the history of episodes
        episodic_reward_history.append(episode_reward)

        # Displaying the total reward for every 20 iterations
        if i % interval == 0:
            print(f'Episode {i} Last episode Reward = {episode_reward} Running reward = {round(running_reward, 2)}')

    env.close()

    # Saving the scores to file
    f = open(f'./Results/scores_record_{num_episodes}_{gamma}_{actor_lr}_{critic_lr}', 'wb')
    pickle.dump(episodic_reward_history, f)

    # Saving the actor and critic parameters
    actor_param_path = open(f'./Results/actor_train_param_{num_episodes}_{gamma}_{actor_lr}', 'wb')
    critic_param_path = open(f'./Results/critic_train_param_{num_episodes}_{gamma}_{critic_lr}', 'wb')
    torch.save(model.actor.state_dict(), actor_param_path)
    torch.save(model.critic.state_dict(), critic_param_path)

    # Plotting the smooth rewards
    smoothed_rewards = pd.Series.rolling(pd.Series(episodic_reward_history), 50).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(smoothed_rewards)
    plt.xlabel('Number of episodes')
    plt.ylabel('Total Episodic Reward')
    plt.savefig('./Results/figure')
    plt.show()
