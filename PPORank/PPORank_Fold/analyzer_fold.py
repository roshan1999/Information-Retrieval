import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt


results = {}
for fold in [5]:
	with open('./Result/MQ2008/Fold'+str(fold)+'/test_i_fold_'+ str(fold) +'_iter_50_clip_0.1_c1_0.5_c2_0.01_lambda_0.95_actor_lr_9e-05_critic_lr_0.0001_g_1.0_hnodes_45_T_20_mbsize_5_epochs_1','rb') as handle:
		result=pickle.load(handle)
		results[fold] = []
		results[fold] = result
		print(results[fold][-1])
		print(f"Fold {fold} best scores: ")
		for val in [0,2,4,9]:
			val_score = [k[0][val] for k in results[fold]]
			print(f"Maximum for NDCG {val} = {np.max(val_score)} at iteration index = {np.argmax(val_score)}")

# results["mean"] = {}
# results["mean"]["1"] = []
# results["mean"]["3"] = []
# results["mean"]["5"] = []
# results["mean"]["10"] = []
# for i in range(0,len(results[1])):
# 	NDCGk = 0
# 	NDCG1 = 0
# 	NDCG3 = 0
# 	NDCG5 = 0
# 	NDCG10 = 0
	

# 	for j in range(1,6):
# 		curr_result = results[j][i]
# 		NDCG1 += curr_result[0][0]
# 		NDCG3 += curr_result[0][2]
# 		NDCG5 += curr_result[0][4]
# 		NDCG10 += curr_result[0][9]

# 	results["mean"]["1"].append(NDCG1/5)
# 	results["mean"]["3"].append(NDCG3/5)
# 	results["mean"]["5"].append(NDCG5/5)
# 	results["mean"]["10"].append(NDCG10/5)


# for i in results["mean"]:
# 	print("\nInitial value for NDCG @",i,results["mean"][i][0])
# 	print("\nFinal Value for NDCG @",i, results["mean"][i][-1])


# for k in ["1","3","5","10"]:
# 	iteration = np.argmax(results["mean"]["1"])
# 	print(f"Iteration with Max value for NDCG@{k} = {iteration}")
# 	for i in results["mean"]:
# 		print("\nInitial value for NDCG @",i,results["mean"][i][0])
# 		print("\nFinal Value for NDCG @",i, results["mean"][i][iteration])


