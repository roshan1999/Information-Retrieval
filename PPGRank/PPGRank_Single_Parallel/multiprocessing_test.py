import multiprocessing as mp
import numpy as np
from Agent import total_return, score
from LoadData import import_dataset
from Test import test
from Sample import sample_episode, sample_episode_test
from Evaluation import validate, validate_individual
from View import display
import time


# Parallel implementation with queries Q

# input: 
# Q = query,
# data = {query: {doc: [feature, label]}}
# gamma = 1
# action_taken = dictionary of action taken for query q
# return: 
# delta_theta (only training)
# action_taken

class Ranker:
    data = {}
    theta = []
    gamma = 1
    epochs = 1
    n_features = 45
    disp = False
    alpha = 0.005

    def __init__(self, training_set, theta, epochs, n_features, gamma, alpha, disp):
        self.data = import_dataset(training_set, n_features)
        self.theta = theta
        self.gamma = gamma
        self.n_features = n_features
        self.epochs = epochs
        self.disp = disp
        self.alpha = alpha

    def single_query(self, Q):
        S = [Q, self.data[Q]]
        M = len(S[1])
        action_taken = []
        delta_theta = 0
        for t in range(0, M):
            # First trajectory
            E_a = sample_episode(self.theta, S)
            # Long term return of E_a = G_a
            G_a = total_return(E_a, self.gamma)
            # Computing score function
            score_a = score(E_a, self.theta)
            # Second trajectory
            E_b = sample_episode(self.theta, S)
            # Long term return of E_b = G_b
            G_b = total_return(E_b, self.gamma)
            # Computing score function
            score_b = score(E_b, self.theta)

            # Update change
            delta_theta += (G_a - G_b) * (score_a - score_b)

            # Algorithm says S[t+1] but in the loop this shall
            # cause the State to skip immediate states
            # because of the t.
            # Since E holds the next state as E[1] Hence constant
            # and not t+1

            if t != M - 1:
                if G_a >= G_b:
                    S = E_a[1][0]
                    action_taken.append(E_a[0][1])
                else:
                    S = E_b[1][0]
                    action_taken.append(E_b[0][1])
            else:
                if G_a >= G_b:
                    action_taken.append(E_a[0][1])
                else:
                    action_taken.append(E_b[0][1])

        return [Q, action_taken, delta_theta]

    # Testing call for a query (multiprocessing across queries)
    def single_test_query(self, Q):
        # Start with state S
        S = [Q, self.data[Q]]

        # Total number of documents in state S
        M = len(S[1])

        # List of action taken
        action_taken = []

        # Update Parameter delta theta
        delta_theta = 0

        # For all documents in current state
        for t in range(0, M):

            # First trajectory
            E_a = sample_episode_test(self.theta, S)

            # Long term return of E_a = G_a
            if t!=M-1:
                S = E_a[1][0]
                action_taken.append(E_a[0][1])

        return [Q, action_taken, delta_theta]

    def train(self, validation_set, training_set, testset, num_process):
        iterative_results_train = {}
        iterative_results_valid = {}
        iterative_results_test = {}
        
        for iterations in range(0, self.epochs):
            print(f"\n--Epoch: {iterations}--\n")
            self.data = import_dataset(training_set, self.n_features)
            action_taken = {}
            queries = list(self.data)

            # Performing training
            print("\nTraining.....\n\n")
            print(f"Number of queries = {len(queries)}\n")
            print("Parallel processing--- \n")
            ts = time.time()
            delta_theta = 0

            # Number of processes for multiprocessing
            pool = mp.Pool(processes=num_process)
            results = pool.map(self.single_query, queries)

            # Collecting all results
            for result in results:
                delta_theta += result[2]
                action_taken[result[0]] = result[1]

            # Synchronizing
            pool.close()
            pool.join()

            print(f"Time in parallel = {time.time() - ts}")

            # Updating theta
            self.theta += self.alpha * np.reshape(delta_theta, (self.n_features, 1))

            # Calculates and returns training set's individual query NDCG scores and mean score
            results_train = validate(self.data, action_taken)
            iterative_results_train[iterations] = results_train
            display(iterative_results_train[iterations], mean=True)

            ts = time.time()

            # Calculates and returns test set's individual query NDCG scores and mean score
            self.data = import_dataset(testset, self.n_features)
            action_taken = {}
            queries = list(self.data)
            print("\nTesting.....\n\n")
            print(f"Number of queries = {len(queries)}\n")
            print("Parallel processing--- \n")
            ts = time.time()
            pool = mp.Pool(processes=num_process)
            results = pool.map(self.single_test_query, queries)

            for result in results:
                action_taken[result[0]] = result[1]

            pool.close()
            pool.join()

            print(f"Time in parallel = {time.time() - ts}")

            results_test = validate(self.data, action_taken)
            iterative_results_test[iterations] = results_test
            display(iterative_results_test[iterations], mean=True)


        return self.theta, iterative_results_train, iterative_results_test, 0


def train(validation_set, training_set, testset, theta, epochs, n_features, gamma, alpha, disp, num_process):
    iterRank = Ranker(training_set, theta, epochs, n_features, gamma, alpha, disp)
    return(iterRank.train(validation_set, training_set, testset, num_process))
