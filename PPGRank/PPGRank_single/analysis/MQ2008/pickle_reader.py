import pandas as pd
import pickle
import argparse
import numpy as np

# Initializing
training = {}
validation = {}
testing = {}
training["mean"] = [0] * 5
validation["mean"] = [0] * 5
testing["mean"] = [0] * 5
training["max"] = [0] * 5
validation["max"] = [0] * 5
testing["max"] = [0] * 5


# Taking fold results location
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True)
args = vars(ap.parse_args())
datadir = args["data"]

# print(training)
for i in range(1, 6):
    stri = f'{datadir}/fold_{i}_data'
    f = open(stri, 'rb')
    obj = pickle.load(f)
    for j in range(0, 5):
        training["mean"][j] += obj["Training"][18][j]
        validation["mean"][j] += obj["Validation"][18][j]
        testing["mean"][j] += obj["Test"][18][j]
        training["max"][j] = max(training["max"][j], obj["Training"][-1][j])
        validation["max"][j] = max(validation["max"][j], obj["Validation"][-1][j])
        testing["max"][j] = max(testing["max"][j], obj["Test"][-1][j])

# Performing average
for k in range(0, 5):
    training["mean"][k] /= 5
    validation["mean"][k] /= 5
    testing["mean"][k] /= 5

# Printing the values
print("\nAverage: \n\nTraining\n")
training["mean"] = np.round(training["mean"],4)
print(f"NDCG@1: {training['mean'][1]}\t NDCG@3: {training['mean'][2]}\t NDCG@5: {training['mean'][3]}\t NDCG@10: {training['mean'][4]}")


print("\n\nValidation\n")
validation["mean"] = np.round(validation["mean"],4)
print(f"NDCG@1: {validation['mean'][1]}\t NDCG@3: {validation['mean'][2]}\t NDCG@5: {validation['mean'][3]}\t NDCG@10: {validation['mean'][4]}")


print("\n\nTesting\n")
testing["mean"] = np.round(testing["mean"],4)
print(f"NDCG@1: {testing['mean'][1]}\t NDCG@3: {testing['mean'][2]}\t NDCG@5: {testing['mean'][3]}\t NDCG@10: {testing['mean'][4]}")


print("\nMax: \n\nTraining\n")
training["max"] = np.round(training["max"],4)
print(f"NDCG@1: {training['max'][1]}\t NDCG@3: {training['max'][2]}\t NDCG@5: {training['max'][3]}\t NDC@G10: {training['max'][4]}")

print("\n\nValidation\n")
validation["max"] = np.round(validation["max"],4)
print(f"NDCG@1: {validation['max'][1]}\t NDCG@3: {validation['max'][2]}\t NDCG@5: {validation['max'][3]}\t NDCG@10: {validation['max'][4]}")


print("\n\nTesting\n")
testing["max"] = np.round(testing["max"],4)
print(f"NDCG@1: {testing['max'][1]}\t NDCG@3: {testing['max'][2]}\t NDCG@5: {testing['max'][3]}\t NDCG@10: {testing['max'][4]}\n")