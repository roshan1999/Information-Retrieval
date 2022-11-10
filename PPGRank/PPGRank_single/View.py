import numpy as np


def display(results, mean=False):
    if mean:
        results["mean"] = np.round(results["mean"], 4)
        print(f"Mean:\n\tNDCG = {results['mean'][0]} NDCG@1 = {results['mean'][1]}"
              f" NDCG@3 = {results['mean'][2]} NDCG@5 = {results['mean'][3]} NDCG@10 = {results['mean'][4]}")
    else:
        for query in results:
            NDCG = results[query][0]
            NDCG_1 = results[query][1]
            NDCG_3 = results[query][2]
            NDCG_5 = results[query][3]
            NDCG_10 = results[query][4]

            if query != "mean":
                print(f"Query = {query} NDCG = {NDCG} NDCG@1 = {NDCG_1}"
                      f" NDCG@3 = {NDCG_3} NDCG@5 = {NDCG_5} NDCG@10 = {NDCG_10}")
                print()
            else:
                print(f"Mean:\n NDCG = {NDCG} NDCG@1 = {NDCG_1}"
                      f" NDCG@3 = {NDCG_3} NDCG@5 = {NDCG_5} NDCG@10 = {NDCG_10}")
