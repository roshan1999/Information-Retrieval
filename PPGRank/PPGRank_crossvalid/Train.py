import numpy as np
from Agent import total_return, score
from LoadData import import_dataset
from Sample import sample_episode
from Evaluation import validate, validate_individual
from View import display


def train(training_set, theta, epochs, n_features, gamma, alpha, disp):
    iterative_results_train = {}

    for iterations in range(0, epochs):
        print(f"Iteration number: {iterations}")
        delta_theta = 0
        data = import_dataset(training_set, n_features)
        action_taken = {}
        queries = list(data)
        print("\n\nTraining.....\n\n")
        print(f"Number of queries = {len(queries)}")
        for Q in queries:
            S = [Q, data[Q]]
            M = len(S[1])
            action_taken[Q] = []
            for t in range(0, M):

                # First trajectory
                E_a = sample_episode(theta, S)
                # Long term return of E_a = G_a
                G_a = total_return(E_a, gamma)
                # Computing score function
                score_a = score(E_a, theta)
                # Second trajectory
                E_b = sample_episode(theta, S)
                # Long term return of E_b = G_b
                G_b = total_return(E_b, gamma)
                # Computing score function
                score_b = score(E_b, theta)

                # Update change
                delta_theta += (G_a - G_b) * (score_a - score_b)

                """The following is to be verified"""
                # Algorithm says S[t+1] but in the loop this shall
                # cause the State to skip immediate states
                # because of the t.
                # Since E holds the next state as E[1] Hence constant
                # and not t+1

                if t != M - 1:
                    if G_a >= G_b:
                        S = E_a[1][0]
                        action_taken[Q].append(E_a[0][1])
                    else:
                        S = E_b[1][0]
                        action_taken[Q].append(E_b[0][1])
                else:
                    if G_a >= G_b:
                        action_taken[Q].append(E_a[0][1])
                    else:
                        action_taken[Q].append(E_b[0][1])

            if disp:
                print("Query = " + str(Q))
                validate_individual(data[Q], action_taken[Q], display=True)
                print()

        theta += alpha * np.reshape(delta_theta, (n_features, 1))

        # Calculates and returns training set's individual query NDCG scores and mean score
        results_train = validate(data, action_taken)
        iterative_results_train[iterations] = results_train
        display(iterative_results_train[iterations], mean=True)

    return theta, iterative_results_train
