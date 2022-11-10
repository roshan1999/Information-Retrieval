import pickle 
import matplotlib.pyplot as plt
import argparse
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument("-d","--datadir",required=True,help="Path to data file to visualize")
args=vars(ap.parse_args())
path=str(args["datadir"])


with open(path,'rb') as handle:
	results=pickle.load(handle)

lst_returns=[]
lst_avgreturns=[]
lst_step=[]
for i in results['training']:
    lst_returns.append(i[-2])
    lst_avgreturns.append(i[-1])
    lst_step.append(i[-3])

#skip first entry since no initial reward available
lst_step=lst_step[1:]
#Smooth the curve
smooth_returns = pd.Series.rolling(pd.Series(lst_step), 10).mean()
#Plot
plt.title("IncrementalRank(Linear)")
plt.plot(lst_step, markersize='5', color='blue', label='Avg Step Reward')
plt.plot(smooth_returns, linewidth=3,markersize='5', color='green', label='Smooth Avg Step Reward')
plt.ylabel("Avg Step Reward")
plt.xlabel("Epochs")
plt.legend()
plt.grid()
plt.show()