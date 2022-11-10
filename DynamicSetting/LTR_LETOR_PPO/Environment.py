import math
from LoadData import import_dataset, import_all


class Dataset:
    def __init__(self, dataset):
        self.QUERY_TRAIN, self.QUERY_TEST, self.QUERY_DOC, self.QUERY_DOC_TRUTH, \
            self.DOC_REPR, self.MAX_DCG = transform_all(dataset)

    def getTrain(self):
        return self.QUERY_TRAIN

    def getTest(self):
        return self.QUERY_TEST

    def getDocQuery(self):
        return self.QUERY_DOC

    def getTruth(self):
        return self.QUERY_DOC_TRUTH

    def getFeatures(self):
        return self.DOC_REPR

    def getIDCG(self):
        return self.MAX_DCG

    def getLabels(self, Q, docLst):
        return [self.QUERY_DOC_TRUTH[Q][x] for x in docLst]


def transform_all(dataset):
    train_data, test_data = import_all(dataset)
    all_data = {**train_data, **test_data}


    # List of queries
    QUERY_TRAIN = list(train_data)
    QUERY_TEST = list(test_data)

    # All queries and docs dictionary
    QUERY_DOC = {}
    DOC_REPR = {}
    QUERY_DOC_TRUTH = {}
    for query in list(all_data):
        QUERY_DOC[query] = list(all_data[query])
        QUERY_DOC_TRUTH[query] = {}
        for doc in list(all_data[query]):
            DOC_REPR[doc] = all_data[query][doc][0]
            QUERY_DOC_TRUTH[query][doc] = all_data[query][doc][1]

    # IDCG
    MAX_DCG = {}
    labels = {}
    for query in list(QUERY_DOC_TRUTH):
        MAX_DCG[query] = []
        labels[query] = []
        for doc in list(all_data[query]):
            labels[query].append(all_data[query][doc][1])
        for doc_pos in range(1, 11):
            if len(labels[query]) >= doc_pos:
                MAX_DCG[query].append(DCG(sorted(labels[query], reverse=True), doc_pos))
            else:
                MAX_DCG[query].append(0)

    return QUERY_TRAIN, QUERY_TEST, QUERY_DOC, QUERY_DOC_TRUTH, DOC_REPR, MAX_DCG


def get_reward(t, Y_at):
    """Calculates reward at time t given label Y_at """
    if t == 0:
        return (2 ** Y_at) - 1
    else:
        return ((2 ** Y_at) - 1) / float(math.log((t + 1), 2))

def DCG(Y, k=None):
    """Calculates DCG at position k"""
    if k is None:
        k = len(Y)
    val = 0
    for i in range(k):
        val += (((2 ** (Y[i])) - 1) / (math.log(i + 2, 2)))
    return val

def update_state(t, doc_action, state):
    """Updates the state based on action"""

    X = []
    for items in state:
        if items != doc_action:
            X.append(items)

    # X = all documents
    # return state as documents
    return X