import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from rank import getcontent
from scoreGen import processLite
import math


# Calculate reward on the basis of the relevance score
def single_reward(score, index):
    if index == 0:
        return (2 ** score) - 1
    else:
        return ((2 ** score) - 1) / float(math.log((index + 1), 2))


# Using random distribution to sample a document
def policy_categorical(W, X):
    """Given Policy Weights W and Feature Vectors of all docs X, returns index of best document"""

    # Calculating softmax policy
    features = np.array(X)
    Xi = np.matmul(features, W)

    # Xi is the data for (theta)*phi <-- generated for softmax
    Xi_tensor = torch.from_numpy(Xi)  # Convert to tensor

    softmax = nn.Softmax(dim=0)  # Softmax function
    output = softmax(Xi_tensor)  # Run through softmax

    # Probability of choosing that action with all documents
    pi = output.reshape(len(output))

    # Performing categorical distribution
    m = Categorical(pi)

    return m.sample()  # Get the index for a trajectory


# Sample an entire trajectory of episode
def sample_episode_Categorical(W, concat_fv_q):

    M = len(list(concat_fv_q)) # Length of the list of documents
    new_doc_list = []

    doc_list = list(concat_fv_q) # Dict of {docIds: fv} for query Q

    episode = [] # Buffer for storing all episodes

    # for each time step, choose an action, update state (by removing chosen document)
    for t in range(M):
        features_list = []

        # Choose an action
        for docId in doc_list:
            features_list.append(concat_fv_q[docId])
        at = policy_categorical(W, features_list)
        episode.append(([t, doc_list], doc_list[at]))
        new_doc_list = []
        
        # Update state by removing document
        for docId in doc_list:
            if doc_list[at] != docId:
                new_doc_list.append(docId)
        doc_list = new_doc_list.copy()
    
    return episode


# Fetching the vectors from the model
def generate_doc_vectors(docLst, docoffset, model):
    doc_fv = {}
    with open("msmarco-docs.tsv", 'r', encoding='utf8') as f:
        for docId in docLst:
            doc_fv[docId] = []
            try:
                v2 = model.infer_vector(processLite(getcontent(docId, docoffset, f)[3]))
                doc_fv[docId] = v2
            except:
                print(f'DociD:{docId} skipped')
    return doc_fv



"""Min Max Scaling done on input and returned"""
def normalize_minmax(features):
    if np.max(features) - np.min(features) == 0:
        return features
    return (features - np.min(features)) / (np.max(features) - np.min(features))
    

# Starting with random weights
def initialize_weights(n_features):
    # Initialize policy weights:
    W = np.random.randn(n_features)*np.sqrt(2/n_features)
    W = np.reshape(W, (n_features, 1))
    W = np.round(normalize_minmax(W) * 2 - 1, 4)
    return W


# Ranking performed by the RL agent
def rank(curr_docLst, curr_query_id, docoffset, model, W):

    # Generate each document's feature vectors
    # doc_fv: {docID: [feature vectors of doc]}
    doc_fv = generate_doc_vectors(curr_docLst, docoffset, model)

    # Sample an episode for ranking and get the actions (ranked list of documents)
    sample_episode = []
    action_list = []
    sample_episode = sample_episode_Categorical(W, doc_fv)
    action_list = [E[1] for E in sample_episode]

    return sample_episode, action_list


# Generating all rewards for the list of documents on screen
def gen_rewards(docLst, doc_M_rs):
    rewards_lst = []
    for index, doc in enumerate(docLst):
        rewards_lst.append(single_reward(doc_M_rs[doc], index))
    return rewards_lst


# Get discounted total returns
def get_returns(rewards, gamma):
    R = 0
    returns = []
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    return returns


# Takes the current state,action,theta and computes the scorefunction
def compute_scorefn(s,a,W,doc_fv):
    # selected features- sum(features*prob(selectfeatures))
    docidsleft=s[1]

    X=[] # feature matrix, features of all docs in the current state.

    for docid in docidsleft:
        X.append(doc_fv[docid])

    X=np.array(X) # (ndocs,nfeatures)
    scores=np.matmul(W.T,X.T).T # (ndocs,1)
    scores=np.round(scores,5)
    probs=np.exp(scores)/np.sum(np.exp(scores)) # softmax

    return np.round(doc_fv[a]-np.sum(X*probs,axis=0),5).reshape(1,len(W))


# Updating the agent's policy. Uses the episode sampled and the feedbacks given to update the policy
def update(W, userFeedbacks, doc_M_rs, query_doc_N_dict, sample_episode, docoffset, model, gamma, n_features, learning_rate):
    # Calculate delta w based on these rewards -- after clicking on update to next timestep
    delta_w=np.zeros(n_features)

    # For each query calculate total delta w
    for q in userFeedbacks:

        # For each state (top 10) calculate return and delta_w
        query_rewards = gen_rewards(query_doc_N_dict[q[1]], doc_M_rs)
        returns = get_returns(query_rewards, gamma)

        E = sample_episode[q[1]]
        curr_doc_list = E[0][0][1]
        curr_doc_fv = generate_doc_vectors(curr_doc_list, docoffset, model)
        for t in range(0, len(list(E))):
            # Current state's return:
            G_t = returns[t]
            # Compute the scorefn
            scorefn_t = compute_scorefn(E[t][0], E[t][1], W, curr_doc_fv)
            # Update Policy Gradient
            delta_w = delta_w + gamma**t * G_t * scorefn_t
    
    # Update policy weights
    W = W + learning_rate * delta_w.reshape(n_features, 1)

    return W