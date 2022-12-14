import numpy as np
from math import sqrt
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

def normalize_std(income):
    """Mean Normalization on input and returned"""
    print(f"Mean = {np.mean(income)} Std = {np.round(np.std(income))}")
    print(f"Mean hand = {mean(income)} STD hand = {std_hand(income)}")
    income -= mean(income)
    print(income)
    std_h = std_hand(income)
    if std_h == 0:
        return np.round(income, 4)
    return np.round((income / std_h), 4)


def normalize_minmax(features):
    """Min Max Scaling done on input and returned"""
    if np.max(features) - np.min(features) == 0:
        return features
    return (features - np.min(features)) / (np.max(features) - np.min(features))


def import_dataset(filename, n_features):
    """Imports file into a structure:{qid:{docid:(features,label),docid2:(features_2,label_2)},qid2:{docid:(),docid2:()}} """
    f = open(filename, "r")
    data = {}
    genid = 0
    for line in f.readlines():
        x = line.split(" ")
        label = int(x[0])
        qid = int(x[1].split(":")[1])
        features = []
        for i in range(n_features):
            features.append(float(x[2 + i].split(":")[1]))
        #         features = normalize(features)
        if "#docid" in x:
            docid = x[x.index("#docid") + 2]
        else:
            docid = str(genid + 1)
            genid += 1
        if qid not in data:
            data[qid] = {docid: (features, label)}
        else:
            data[qid][docid] = (features, label)
    return data


def import_all(filename):
    f = open(filename, 'rb')
    t = pickle.load(f)
    s = pd.Series(t)
    training_data, test_data = [i.to_dict() for i in train_test_split(s, train_size=0.7, random_state=4)]
    return training_data, test_data
