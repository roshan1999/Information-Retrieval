import math


def DCG(Y, k=None):
    """Calculates DCG at position k"""
    val = 0
    for i in range(k):
        val += (((2 ** (Y[i])) - 1) / (math.log(i + 2, 2)))
    return val


def import_dataset(dataset):
    data = {}
    n_features = 45
    for line in dataset.readlines():
        x = line.split(" ")
        label = int(x[0])
        qid = int(x[1].split(":")[1])
        features = []
        genid = 0
        for i in range(n_features):
            features.append(float(x[2 + i].split(":")[1]))
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


def main():
    trainset = open("./data/OHSUMED/data/Fold1/train.txt")
    testset = open("./data/OHSUMED/data/Fold1/test.txt")
    valiset = open("./data/OHSUMED/data/Fold1/vali.txt")

    test = import_dataset(testset)
    train = import_dataset(trainset)
    vali = import_dataset(valiset)

    all_data = {**test, **train, **vali}

    # List of queries
    QUERY_TEST = list(test)
    QUERY_TRAIN = list(train)
    QUERY_VALI = list(vali)

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
        for doc_pos in range(10):
            MAX_DCG[query].append(DCG(sorted(labels[query], reverse=True), doc_pos))

    return QUERY_VALI, QUERY_TEST, QUERY_TRAIN, QUERY_DOC, QUERY_DOC_TRUTH, DOC_REPR, MAX_DCG
