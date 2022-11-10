import math


def DCG(Y, k=None):
    """Calculates DCG at position k"""
    if k is None:
        k = len(Y)
    val = 0
    if k > len(Y):
        return k + 1
    for i in range(k):
        val += (((2 ** (Y[i])) - 1) / (math.log(i + 2, 2)))
    return val


def evaluate(E, X, k=None):
    """Calculates NDCG@k"""
    labels = [X[at[1]][1] for at in E]
    IDCG = DCG(sorted(labels, reverse=True), k)
    if IDCG == 0 or IDCG == -1:
        # print("Can't compute NDCG as IDCG=0, returning DCG")
        return 0
    NDCG = DCG(labels, k) / IDCG
    return NDCG


def validate_individual(data_query, action_taken, display=False):
    label = []
    for action in action_taken:
        label.append(data_query[action][1])

    IDCG = DCG(sorted(label, reverse=True))

    if IDCG == 0:
        # IDCG = 0, hence can't compute NDCG
        results = [0] * 5
        if display:
            print("IDCG = 0")
        return results

    DCG_cal = {}
    results = [DCG(label) / IDCG]
    for k in [1, 3, 5, 10]:
        DCG_cal[k] = DCG(label, k)
        if DCG_cal[k] == k + 1:
            results.append(0)
        else:
            results.append(DCG_cal[k] / DCG(sorted(label, reverse=True), k))

    if display:
        print(f"NDCG = {round(results[0], 4)} NDCG@1 = {round(results[1], 4)}"
              f" NDCG@3 = {round(results[2], 4)} NDCG@5 = {round(results[3], 4)} NDCG@10 = {round(results[4], 4)}")

    return results


def validate(data, action_taken):
    results = {}
    mean_results = [0, 0, 0, 0, 0]

    for query in action_taken:
        results[query] = validate_individual(data[query], action_taken[query])

        for value in range(len(mean_results)):
            mean_results[value] += results[query][value]

    for value in range(len(mean_results)):
        mean_results[value] /= len(action_taken)

    results["mean"] = mean_results

    return results
