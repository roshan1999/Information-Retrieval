import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def get_reward(t, Y_at):
    """Calculates reward at time t given label Y_at """
    if t == 0:
        return (2 ** Y_at) - 1
    else:
        return ((2 ** Y_at) - 1) / float(math.log((t + 1), 2))


# Probability of selecting a document
def policy(theta, features):
    """Given Policy Weights W and Feature Vectors of all docs X, returns index of best document"""

    # Calculating softmax policy
    features = np.array(features)
    Xi = np.matmul(features, theta)

    # Xi is the data for (theta)*phi <-- generated for softmax

    Xi_tensor = torch.from_numpy(Xi)  # Convert to tensor

    softmax = nn.Softmax(dim=0)  # Softmax function
    output = softmax(Xi_tensor)  # Run through softmax

    # Output shape = 188,1 => Probability of choosing that action with all documents
    pi = output.reshape(len(output))

    # Performing categorical distribution
    m = Categorical(pi)

    return m.sample()  # Get the index for a trajectory


def sample_episode(theta, S):
    """Samples an episode and returns E = [(S_t, A_t, R_t+1), ..]"""
    X = S[1]  # {docID: (features, labels)}
    Q = S[0]  # Query
    M = len(X)
    E = []

    # copied to Y to allow immutable deletion
    Y = X

    for t in range(M):
        features = []  # To store features in the same order as DocID
        labels = []  # To store relevance in the same order as DocID

        for docId in Y:
            features.append(Y[docId][0])
            labels.append(Y[docId][1])

        at = policy(theta, features)  # at = index of docID
        action_doc = list(Y)[at.item()]
        rt_1 = get_reward(t, labels[at])
        E.append(([Q, Y], action_doc, rt_1))  # E = (S,A_t,R_t+1)

        # Deleting action from list of documents
        X = Y
        Y = {}
        for items in X:
            if action_doc != items:
                Y[items] = X[items]
    return E
