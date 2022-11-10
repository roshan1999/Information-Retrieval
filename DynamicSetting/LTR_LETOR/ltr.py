from gensim.models.doc2vec import Doc2Vec
import random
import time
import pickle
import pandas as pd
random.seed(6)


# Generate M documents and initialize with RS = 1
def generateFromLst(data, M):
    doc_M_rs = {}
    doc_query = data.getDocQuery()
    for q in doc_query:
        for doc in doc_query[q]:
            doc_M_rs[doc] = 1
            if len(doc_M_rs)==M:
                return doc_M_rs
    print("M exceeded size of corpus")
    exit()


# Retrieve documents based on Relevance Label and then Relevance score 
def retrieve(qid, doc_M_rs, data, k):
    query_doc_list = data.getDocQuery()[qid]
    retrieve_list = []
    for doc in query_doc_list:
        if doc in list(doc_M_rs) and doc not in retrieve_list:
            retrieve_list.append(doc)
            if len(retrieve_list) == k:
                return retrieve_list

    doc_M_rs = dict(sorted(doc_M_rs.items(), key=lambda item: item[1]))
    for doc in doc_M_rs:
        if doc not in retrieve_list:
            retrieve_list.append(doc)
            if len(retrieve_list) == k:
                return retrieve_list


# Function to generate user feedback on the basis of the relevance label of the query
def generate_userFeedback(queryId, curr_docList, number_users, data):
    # Relevance Labels of all docs fetched
    doc_labels = data.getTruth()[queryId]   

    curr_doc_labels = {}
    for index, doc in enumerate(curr_docList):
        if doc in doc_labels:
            if doc_labels[doc]!=0: # Dont include documents with 0 relevance labels
                curr_doc_labels[doc] = [index+1, doc_labels[doc]]

    # Generating automatic user feedback for query
    sortedLabels = dict(sorted(curr_doc_labels.items(), key=lambda item: item[1][1], reverse=True))
    userfeedback = {}

    for user in range(0, min(number_users, len(sortedLabels))):
        userfeedback[list(sortedLabels)[user]] = sortedLabels[list(sortedLabels)[user]][0]
    # print("Automatic feedbacks ")
    # print(userfeedback)
    return userfeedback


# Add to relevance score with (rel_label * 2/position) of document
def updateRelevanceScore(doc_M_dict, userfeedbacks, decay_value, data):

    doc_query_labels = data.getTruth()# {query:{doc:label}}
    for doc in doc_M_dict:
        flag = 0
        for q in userfeedbacks:
            if q[0] == 1: # if manual feedback, give highest relevance label 
                if doc in list(userfeedbacks[q]):
                    doc_M_dict[doc] += (4/userfeedbacks[q][doc])
                    flag = 1
            elif q[0] == 0:
                if doc in list(userfeedbacks[q]):
                    doc_M_dict[doc] += (doc_query_labels[q[1]][doc] * 2/userfeedbacks[q][doc])
                    flag = 1
        if flag == 0:
            # Decay only if value is above 0
            if doc_M_dict[doc] > 0:
                doc_M_dict[doc] -= decay_value

    return doc_M_dict


# Function to perform sliding window
def slideWindow(doc_M_rs, window_size):

    # Given a fixed size to window
    if len(doc_M_rs) >= window_size:
        print("Maximum Window size reached, removing documents")
        delLst = list(dict(sorted(doc_M_rs.items(), key=lambda item: item[1])))[0:(len(doc_M_rs) - window_size)]
        print(len(delLst))
        # Remove documents until window size reached
        for doc in delLst:
            del doc_M_rs[doc]

        # Remove documents with RS <=0
        for doc in list(doc_M_rs):
            if doc_M_rs[doc] <= 0 and len(doc_M_rs) > 40:
                del doc_M_rs[doc]
        print(f"Total Documents in system: {len(doc_M_rs)}")
    else:
        print("Maximum window size not reached")
    return doc_M_rs

# Function to add documents. 
def extendDocList(doc_M_rs, N, data):
    M = len(list(doc_M_rs))
    print(f'Len before adding documents: {M}')
    iterate = 0
    total_docs = len(data.getFeatures())
    if (M + N) > total_docs:
        print("Adding more than the number of documents available in corpus")
        print("Correcting to max possible")
        N = len(data.getFeatures()) - M
    all_doc_query = data.getDocQuery()
    for q in all_doc_query:
        for doc in all_doc_query[q]:
            if doc not in list(doc_M_rs):
                doc_M_rs[doc] = 1
                iterate+=1
                if iterate == N:
                    print(f'Len after adding documents: {len(list(doc_M_rs))}')
                    return doc_M_rs
    if iterate<N:
        print(f"Total documents exceeded size of the corpus\n Final Length = {len(doc_M_rs)}")
        return doc_M_rs

