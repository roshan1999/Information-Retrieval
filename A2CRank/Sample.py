import math
import numpy as np
import torch
import torch.nn as nn


def get_reward(t, Y_at):
    """Calculates reward at time t given label Y_at """
    if t == 0:
        return (2 ** Y_at) - 1
    else:
        return ((2 ** Y_at) - 1) / float(math.log((t + 1), 2))


# Probability of selecting a document
def policy(W, X):
    """Given Policy Weights W and Feature Vectors of all docs X, returns index of best document"""

    # X = ALl the documents of the query
    # Calculating softmax policy
    Xm_values = np.array(X)  # shape = (len(X), 25)
    Xi = np.matmul(Xm_values, W)  # W_shape =(25,1)
    # Xi shape = (188,1) <-- Xi is the data for XW <-- generated for softmax
    Xi_tf = torch.from_numpy(Xi)  # Convert to tensor
    softmax = nn.Softmax(dim=0)  # Softmax function
    output = softmax(Xi_tf)  # Run through softmax
    # Output shape = 188,1 => Probability of choosing that action with all documents
    # return docID with best action
    return torch.argmax(output)  # Get the index with best probability


def sample_episode(W, X):
    """Given Policy Weights W and data dictionary:data[qid] X, returns a episode in the structure:
        E=[([t1,docids_left],docid_selected,rt_1),([s],doc_selected,r),()] along with corresponding labels of docs selected"""

    M = len(X)
    E = []
    Label = []
    for t in range(M):
        X_docID = X.keys()  # List of DocIds
        Xm_values = []  # To store features in the same order as DocID
        Ym_values = []  # To store relevance in the same order as DocID
        for docId in X_docID:
            Xm_values.append(X[docId][0])
            Ym_values.append(X[docId][1])
        at = policy(W, Xm_values)  # at = index of docID
        rt_1 = get_reward(t, Ym_values[at])
        #         print(" t = "+str(t) + " action = " + str(list(X_docID)[at]) + " Reward = " + str(rt_1))
        E.append(([t, list(X)], list(X_docID)[at], rt_1))
        # print("Action chosen with Index = " + str(at.item()) + " DocId = " + str(list(X_docID)[at]))
        # print("Label = " + str(Ym_values[at.item()]))
        Label.append(Ym_values[at.item()])
        if list(X_docID)[at] in X:
            X.pop(list(X_docID)[at.item()])
        else:
            print("DocID doesnt exist " + str(list(X_docID)[at]))
    return E, Label

def sample_episode_imitation(W,X):
    """Takes current weight and feature dictionary {docid:(features,label),docid2:(features2,label2)} 
    and returns the best possible training episode E=[(S(t,docidsleft),A(docidselected),R)] and labels of docs selected """
    #print("Imitation Episode generating")
    lst_docids=[]
    lst_labels=[]
    for docid in X.keys():
        lst_docids.append(docid)
        lst_labels.append(X[docid][1])
    arr_labels=np.array(lst_labels)
    arr_docids=np.array(lst_docids)
    indexs=np.argsort(arr_labels)[np.arange(len(arr_labels)-1,-1,-1)]
    M=len(X)#132 docs
    arr_docids=arr_docids[indexs]
    arr_labels=arr_labels[indexs]
    #Making the episode
    E=[]
    for t in range(M):
        S=[t,list(arr_docids)]
        A=arr_docids[0]
        R=get_reward(t,arr_labels[t])
        E.append((S,A,R))
        arr_docids=np.delete(arr_docids,0)
    return E,list(arr_labels)