from rank import main
import scoreGen as sg
from gensim.models.doc2vec import Doc2Vec
import random
import time
import pickle
import pandas as pd

random.seed(6)


# Generate M docs from msmarco dataset
def generate_M_docs(M, model, corpus_size):
    df = pd.read_csv('./msmarco-docs.tsv', delimiter = "\t", header=None, nrows=corpus_size)
    docdict = {}
    docLst = list(df[0])
    iterate = 0
    for index, doc in enumerate(docLst):
        if doc in model.dv:
            docdict[doc] = 1
            iterate+=1
            if iterate == M:
                return docdict, docLst
    if(iterate<M):
        print("Initial size greater than corpus size")
        exit()


# LOADING THE MODEL
def LoadModel(modelString):
    # For each of the document in the document list for each query use doc2vec
    print("\n----Loading the trained Doc2Vec model----\n")
    model = Doc2Vec.load(modelString)
    # trainAccuracy()
    print("\n---Finished Loading the model ---\n")
    return model


# Generate retrieval function for current Query
def query_retrieval(currentQuery, curr_DocList, alpha, beta, model):
    
    # Similarity score list
    print("\n---Using Doc2Vec model to calculate Similarity Scores for")
    best_docId, ss_list = sg.testModel(currentQuery, list(curr_DocList), model)

    # Setting a retrieval function based on RS and SS
    retrievedList = {} 
    for docId in ss_list:
        retrievedList[docId] = alpha * ss_list[docId] + beta * curr_DocList[docId]

    # Sort the list
    sortedretrievedList = dict(sorted(retrievedList.items(), key=lambda item: item[1], reverse=True))
    return sortedretrievedList


# Generate automatic feedbacks
def generate_userFeedback(currentQuery, curr_DocList, number_users, model):
    # Relevance score list for user feedback
    docid_best_rs, rankedList_rs = sg.testModelRelevance(currentQuery, curr_DocList, model)

    # Generating automatic user feedback for query
    sortedRs = dict(sorted(rankedList_rs.items(), key=lambda item: item[1][0], reverse=True))
    userfeedback = {}

    # Taking min of num_users, len(sortedRs) as some documents might have no data
    for user in range(0, min(number_users, len(sortedRs))):
        userfeedback[list(sortedRs)[user]] = sortedRs[list(sortedRs)[user]][1]

    return userfeedback


# Updating the relevance scores based on the feedbacks given
def updateRelevanceScore(doc_M_dict, userfeedbacks, decay_value):
    for doc in doc_M_dict:
        # Flag indicates a positive user feedback, else decayed
        flag = 0
        for q in userfeedbacks:
            if doc in list(userfeedbacks[q]):
                doc_M_dict[doc] += (1/userfeedbacks[q][doc])
                flag = 1
        if flag == 0:
            if doc_M_dict[doc] > 0:
                doc_M_dict[doc] -= decay_value
    return doc_M_dict


# Perform sliding window
def slideWindow(doc_M_dict, window_size):
    # Given a fixed size to window
    if len(doc_M_dict) >= window_size:
        # Remove documents by worst relevance scores first
        print("Maximum Window size reached, removing documents")
        delLst = list(dict(sorted(doc_M_dict.items(), key=lambda item: item[1])))[0:(len(doc_M_dict) - window_size)]
        print(len(delLst))

        for doc in delLst:
            del doc_M_dict[doc]

        # Post window size capacity achieved, remove documents with relevance score below 0
        for doc in list(doc_M_dict):
            if doc_M_dict[doc] <= 0 and len(doc_M_dict) > 40:
                del doc_M_dict[doc]
        print(f"Total Documents in system: {len(doc_M_dict)}")

    else:
        print("Maximum window size not reached")
    return doc_M_dict


# Add N more documents to the total documents available
def extendDocList(doc_M_dict, N, model, df_lst):
    M = len(list(doc_M_dict))
    print(f'Len before adding documents: {M}')
    iterate = 0
    for index, doc in enumerate(df_lst[M-10:]):
        if doc in model.dv and doc not in list(doc_M_dict):
            doc_M_dict[doc] = 1
            iterate+=1
            if iterate == N:
                print(f'Len after adding documents: {len(list(doc_M_dict))}')
                return doc_M_dict
    if iterate<N:
        print("Corpus exhausted")
        exit()

