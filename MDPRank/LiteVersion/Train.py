import numpy as np
import csv
import time

from Eval import evaluate
from LoadData import import_dataset, normalize_std, normalize_minmax
from Sample import *
from Test import Test


def compute_scorefn(state, action, w, qid, data):
    """Given state,action,weights,qid,data dictionary computes the score function """

    # docids remaining in current state
    X = state[1]
    # x_m(a_t):features of selected document
    arr_selected_features = np.array(data[qid][action][0])
    # Initialize
    numerator, denominator = 0, 0
    # Compute gradient
    for docid in X:
        x_docid = np.round(np.array(data[qid][docid][0]), 4)
        x_p = np.round(np.matmul(w.T, x_docid), 4)
        numerator += np.round(x_docid * np.exp(x_p), 4)
        denominator += np.round(np.exp(x_p), 4)
    # print(denominator)
    if denominator == 0:
        return arr_selected_features
    grad = arr_selected_features - (numerator / denominator)
    return np.round(grad, 4)

def compute_scorefn_v2(s,a,theta,q,data):
    """Takes the current state,action,theta and computes the scorefunction """
    #selected features- sum(features*prob(selectfeatures))
    docidsleft=s[1]
    X=[]#feature matrix, features of all docs in the current state.
    for docid in docidsleft:
        X.append(data[q][docid][0])
    X=np.array(X)#(ndocs,nfeatures)
    scores=np.matmul(theta.T,X.T).T # (ndocs,1)
    scores=np.round(scores,5)
    probs=np.exp(scores)/np.sum(np.exp(scores))#softmax
    return np.round(data[q][a][0]-np.sum(X*probs,axis=0),5).reshape(1,len(theta))

def Train(trainset,validset,testset,n_features, W,epochs,alpha):
    """Trains dataset for given number of iterations using W as starting point and validates after each epoch"""

    #Hyperparameters
    gamma = 1
    learning_rate = alpha
    n_queries=100
    len_episode=999999
    #Import dataset
    data = import_dataset(trainset,n_features)

    #Results dictionary: Has train mean_ndcgs ,validation,test mean_ndcgs across the current fold
    results={'training':[],'validation':[],'test':[]}

    #Main training loop
    for i in range(1,epochs+1):
        #Monitor the start time
        t_start=time.time()

        #Randomize the queries
        queries_list=np.random.choice(list(data.keys()),size=n_queries)

        # print("\n***********EPOCH: " + str(i) + " START***********\n")
        #Monitors
        delta_w = 0
        NDCG_values = []
        NDCG_1_values = []
        NDCG_3_values = []
        NDCG_5_values = []
        NDCG_10_values = []

        #buffer initialized
        delta_w=np.zeros(n_features)
        for q in queries_list:

            X = data[q]  # {docid:(features,labels)}
            M = len(X)  # |x|
            X_copy=X.copy()
            
            #Sample an episode either categorically or best possible episode
            #E, Label_list = sample_episode_categorical(W, X_copy)
            E, Label_list = sample_episode_imitation(W, X_copy)


            # Calculates evaluation measures
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

            # computes the t-step return(starts at t till the end)
            
            for t in range(min(M, len_episode)):
                G_t = 0
                # Get total Return
                for k in range(0, M - t):
                    G_t = G_t + gamma ** (k) * E[t + k][2]
                # print("\tTotal Return = " + str(G_t))
                # Compute the scorefn
                scorefn_t = compute_scorefn_v2(E[t][0], E[t][1], W, q, data)
                # Update Policy Gradient
                delta_w = delta_w + gamma**t * G_t * scorefn_t

        #Updating Policy Weights and Normalizing
        delta_w=delta_w/np.linalg.norm(delta_w)
        W = W + learning_rate * delta_w.reshape(n_features, 1)
        W = W/np.linalg.norm(W)

        # After completion of all queries:Mean NDCGs
        NDCG_mean = np.mean(np.array(NDCG_values))
        NDCG_1_mean = np.mean(np.array(NDCG_1_values))
        NDCG_3_mean = np.mean(np.array(NDCG_3_values))
        NDCG_5_mean = np.mean(np.array(NDCG_5_values))
        NDCG_10_mean = np.mean(np.array(NDCG_10_values))

        #Updating results dictionary
        results['training'].append((NDCG_mean, NDCG_1_mean, NDCG_3_mean,NDCG_5_mean, NDCG_10_mean))

        #Printing Training Scores on the terminal
        print("Training Scores:")
        print(
            f"NDCG mean value = {NDCG_mean} NDCG@1 mean value = {NDCG_1_mean} NDCG@3 mean value = {NDCG_3_mean} NDCG@5 mean value = {NDCG_5_mean} NDCG@10 mean value = {NDCG_10_mean}")
        
        
        # # Printing Validation Scores
        # print("Validation Scores:")
        # ndcg_values = Test(W, validset,n_features)
        # results['validation'].append(tuple(ndcg_values))

        # # Printing Testing Scores
        print("Test Scores:")
        ndcg_values = Test(W, testset,n_features)
        print(f"NDCG mean value = {ndcg_values[0]} NDCG@1 mean value = {ndcg_values[1]} NDCG@3 mean value = {ndcg_values[2]} NDCG@5 mean value = {ndcg_values[3]} NDCG@10 mean value = {ndcg_values[4]}")
        results['test'].append(tuple(ndcg_values))
        
        print("time-taken:",np.round(time.time()-t_start,4),"s")
        print("***********EPOCH: " + str(i) + " COMPLETE***********\n")
        
    return W,results