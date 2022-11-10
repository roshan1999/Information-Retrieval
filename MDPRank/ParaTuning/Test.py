from Eval import evaluate
from LoadData import import_dataset
from Sample import sample_episode
import numpy as np
import csv


def Test(W, dataset,n_features):
    """Given Policy Weights W and test dataset evaluates NDCG values"""

    data = import_dataset(dataset,n_features)
    queries_list = list(data)
    NDCG_values = []
    NDCG_1_values = []
    NDCG_3_values = []
    NDCG_5_values = []
    NDCG_10_values = []
    for q in queries_list:
        # print("Query : " + str(q))
        X = data[q]  # {docid:(features,labels)}
        M = len(X)  # |x|
        E,_ = sample_episode(W, X)
        # Calculates and prints evaluation measures
        data = import_dataset(dataset,n_features)  # importing again
        X = data[q]
        # Calculating reward:
        Gt = 0
        for t in range(M):
            Gt += E[t][2]
        # print(f"Total Reward = {Gt}, Average Reward = {Gt / M}")

        # Calculating NDCG scores
        NDCG = evaluate(E, X)
        NDCG_1 = evaluate(E, X, 1)
        NDCG_3 = evaluate(E, X, 3)
        NDCG_5 = evaluate(E, X, 5)
        NDCG_10 = evaluate(E, X, 10)
        NDCG_values.append(NDCG)
        NDCG_1_values.append(NDCG_1)
        NDCG_3_values.append(NDCG_3)
        NDCG_5_values.append(NDCG_5)
        NDCG_10_values.append(NDCG_10)
        # print(f"NDCG value = {NDCG} @1 = {NDCG_1} @3 = {NDCG_3} @5 = {NDCG_5} @10 = {NDCG_10}")

    NDCG_mean = np.mean(np.array(NDCG_values))
    NDCG_1_mean = np.mean(np.array(NDCG_1_values))
    NDCG_3_mean = np.mean(np.array(NDCG_3_values))
    NDCG_5_mean = np.mean(np.array(NDCG_5_values))
    NDCG_10_mean = np.mean(np.array(NDCG_10_values))
    #print(f"NDCG mean value = {NDCG_mean}")
    #print(f"NDCG mean value = {NDCG_mean} NDCG@1 mean value = {NDCG_1_mean} NDCG@3 mean value = {NDCG_3_mean} NDCG@5 mean value = {NDCG_5_mean} NDCG@10 mean value = {NDCG_10_mean}")
    
    return [NDCG_mean, NDCG_1_mean, NDCG_3_mean, NDCG_5_mean, NDCG_10_mean]
