import math


def DCG(Y, k=None):
    """Calculates DCG at position k"""
    if k is None:
        k = len(Y)
    if(k>len(Y)):
        #print("Y: ",Y,"k:",k)
        #k = len(Y)
        return 0
    val = 0
    for i in range(k):
        val += (((2 ** (Y[i])) - 1) / (math.log(i + 2, 2)))
    return val


def evaluate(E, X, k=None):
    """Calculates NDCG@k"""
    labels = [X[at[1]][1] for at in E]
    IDCG = DCG(sorted(labels, reverse=True), k)
    if IDCG == 0:
        # print("Can't compute NDCG as IDCG=0, returning DCG")
        return 0
    NDCG = DCG(labels, k) / IDCG
    return NDCG
