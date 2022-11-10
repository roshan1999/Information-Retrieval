import pickle
import numpy as np


results=[]
for fold in [1,2,3,4,5]:
	with open('fold_'+str(fold)+'_data','rb') as handle:
		result=pickle.load(handle)
		results.append(result['test'][-1])
#print(results)
print("\nAverage across 5 folds:")
avg_ndcg1,avg_ndcg3,avg_ndcg5,avg_ndcg10=0,0,0,0
for i in range(0,5):
	avg_ndcg1+=results[i][1]/5
	avg_ndcg3+=results[i][2]/5
	avg_ndcg5+=results[i][3]/5
	avg_ndcg10+=results[i][4]/5
print("NDCG@1:",np.round(avg_ndcg1,4),"NDCG@3:",np.round(avg_ndcg3,4),"NDCG@5:",np.round(avg_ndcg5,4),"NDCG@10:",np.round(avg_ndcg10,4),)

print("\nMax across 5 folds:")
max_ndcg1,max_ndcg3,max_ndcg5,max_ndcg10=0,0,0,0
for i in range(0,5):
	max_ndcg1=results[i][1] if(max_ndcg1<results[i][1]) else max_ndcg1
	max_ndcg3=results[i][2] if(max_ndcg3<results[i][2]) else max_ndcg3
	max_ndcg5=results[i][3] if(max_ndcg5<results[i][3]) else max_ndcg5
	max_ndcg10=results[i][4] if(max_ndcg10<results[i][4]) else max_ndcg10
print("NDCG@1:",np.round(max_ndcg1,4),"NDCG@3:",np.round(max_ndcg3,4),"NDCG@5:",np.round(max_ndcg5,4),"NDCG@10:",np.round(max_ndcg10,4),"\n")