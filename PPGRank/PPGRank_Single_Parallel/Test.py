from LoadData import import_dataset
from Sample import sample_episode
from Evaluation import validate, validate_individual
from Agent import total_return


def test(theta, dataset, n_features, gamma, disp):
    """Given Policy Weights theta and test dataset evaluates NDCG values"""
    data = import_dataset(dataset, n_features)
    action_taken = {}
    queries = list(data)
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
            # Second trajectory
            E_b = sample_episode(theta, S)
            # Long term return of E_b = G_b
            G_b = total_return(E_b, gamma)

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

        # if disp:
        #     print("Query = " + str(Q))
        #     # print(action_taken[Q])
        #     validate_individual(data[Q], action_taken[Q], display=True)
        #     print()

    # Calculates and returns individual query NDCG scores

    results = validate(data, action_taken)

    return results


