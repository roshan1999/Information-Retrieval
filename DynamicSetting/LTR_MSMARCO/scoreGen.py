import json
import os
import rank
import re
import gzip
import csv
from nltk import word_tokenize
from nltk.corpus import stopwords
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
import scipy
import numpy as np
import time

import nltk
nltk.download('stopwords')
cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1
stop_words = stopwords.words('english')


# Preprocessing by removing stop words and any symbol
def processLite(txt):
    txt = re.sub(r'[^\w\s]', '', txt)
    # print(text)
    txt = word_tokenize(txt.lower())
    return [word for word in txt if word not in stop_words]


# Calculate the similarity scores of the document list with the query
def testModel(query, vector_id_list, model):
    processed_query = processLite(query)
    print("Query : --- ")
    print(processed_query)
    # Infer query vector
    v1 = model.infer_vector(processed_query)

    # Fetch the document vectors from the model
    data_vec = [model.dv[docid] for docid in vector_id_list]
    maxId = None
    maxCosineSimilarity = 0
    rankedList = {}
    print()

    # Individually find the cosine similarity for each document with respect to the query
    for index, docId in enumerate(vector_id_list):
        try:
            cos_distance = scipy.spatial.distance.cosine(v1, data_vec[index])
            cos_similarity = 1 - cos_distance
            rankedList[docId] = np.round(cos_similarity, 4)
            if maxCosineSimilarity < cos_similarity:
                maxId = docId
                maxCosineSimilarity = cos_similarity
        # print("For DocID: " + docId + " Similarity Score: " + str(cos_similarity))
        except:
            print(f"DocID = {docId} skipped")
            rankedList[docId] = 0
            continue
    # print(f"Most similar doc content: DOCID: {maxId} with cosine similarity = {np.round(maxCosineSimilarity, 4)}\n")
    return maxId, rankedList



# Calculate the similarity scores of the document list 
# with the query for the first 2 lines of the data
def testModelRelevance(query, vector_id_list, model):
    num_lines = 2
    print(f"\n---Generating user feedbacks using first {num_lines} lines of data---\n")
    docoffset = {}

    # Since fetching document hence offset required
    with gzip.open("msmarco-docs-lookup.tsv.gz", 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [docid, _, offset] in tsvreader:
            docoffset[docid] = int(offset)
    
    processed_query = processLite(query)
    print("Query : --- ")
    print(processed_query)
    # Construct the vector of the query
    v1 = model.infer_vector(processed_query)
    maxId = None
    maxCosineSimilarity = 0
    rankedList = {}
    print()

    # Opening msmarco file that will fetch document data based on which SS is calculated
    f = open("msmarco-docs.tsv", 'r', encoding='utf8')

    # Calculate SS based on the first 2 lines of the data
    for index, docId in enumerate(vector_id_list):
        try:
            lst = rank.getcontent(docId, docoffset, f)[3].split('.')[:num_lines]
            if len(lst) < num_lines:
                # print(f"{docId} skipped (<{num_lines})")
                rankedList[docId] = [0, index+1]
                continue
            v2 = model.infer_vector(processLite('.'.join(lst)))
            cos_distance = scipy.spatial.distance.cosine(v1, v2)
            cos_similarity = 1 - cos_distance
            rankedList[docId] = [np.round(cos_similarity, 4), index+1]
            if maxCosineSimilarity < cos_similarity:
                maxId = docId
                maxCosineSimilarity = cos_similarity
            # print("For DocID: " + docId + " Similarity Score: " + str(cos_similarity))
        except:
            print(f"{docId} skipped")
            rankedList[docId] = [0, index+1]
            continue
    f.close()

    # print(f"Most similar doc content based on first {num_lines} lines: DOCID: {maxId} with cosine similarity = {np.round(maxCosineSimilarity, 4)}\n")
    return maxId, rankedList
