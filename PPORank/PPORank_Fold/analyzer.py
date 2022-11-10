import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt


results = {}

ap = argparse.ArgumentParser()
ap.add_argument("-fn", "--filename", required=True, help="File name to average across for folds")
args = vars(ap.parse_args())

fname = str(args["filename"])

for fold in [1,2,3,4,5]:
	with open('./Result/test_fold_'+ str(fold) + fname,'rb') as handle:
		result=pickle.load(handle)
		results[fold] = []
		results[fold] = result

results["mean"] = {}
results["mean"]["1"] = []
results["mean"]["3"] = []
results["mean"]["5"] = []
results["mean"]["10"] = []
for i in range(0,len(results[1])):
	NDCGk = 0
	NDCG1 = 0
	NDCG3 = 0
	NDCG5 = 0
	NDCG10 = 0
	

	for j in range(1,6):
		curr_result = results[j][i]
		NDCG1 += curr_result[0][0]
		NDCG3 += curr_result[0][2]
		NDCG5 += curr_result[0][4]
		NDCG10 += curr_result[0][9]

	results["mean"]["1"].append(NDCG1/5)
	results["mean"]["3"].append(NDCG3/5)
	results["mean"]["5"].append(NDCG5/5)
	results["mean"]["10"].append(NDCG10/5)


# print([results["mean"][i][-1] for i in results["mean"]])
for i in results["mean"]:
	print("\nInitial value for NDCG @",i,results["mean"][i][0])
	print("\nFinal Value for NDCG @",i, results["mean"][i][-1])


# for k in ["1","3","5","10"]:
# 	iteration = np.argmax(results["mean"]["1"])
# 	print(f"Iteration with Max value for NDCG@{k} = {iteration}")
# 	for i in results["mean"]:
# 		print("Initial value for NDCG @",i,results["mean"][i][0])
# 		print("Final Value for NDCG @",i, results["mean"][i][iteration])


