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

lst_avgreturns=[]
for i in results['train']:
    lst_avgreturns.append(i[-1])

#Skip first entry as no starting avgreturn
lst_avgreturns=lst_avgreturns[1:]

#Smooth the returns
smooth_returns = pd.Series.rolling(pd.Series(lst_avgreturns), 10).mean()

fig = plt.figure(figsize=(26, 17))
plt.title("A2C(Linear)")
plt.plot(lst_avgreturns, color='blue', label='Avg Step Reward')
plt.plot(smooth_returns, linewidth=3, color='green', label='Smooth Avg Step Reward')
plt.ylabel("Avg Step Reward")
plt.xlabel("Epochs")
plt.legend()
plt.grid()
plt.show()