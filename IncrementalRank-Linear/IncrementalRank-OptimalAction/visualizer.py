import pickle 
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np

def make_plot(results, axs, pos):
    """Takes the results dictionary, k value as input and plots a graph of NDCG@k on the given axis"""
    #Maps index to ndcg position
    indexer = {1: 1, 2: 3, 3: 5, 4: 10}
    tvals = []
    #vvals = []
    tsvals= []
    for i in results['training']:
        tvals.append(i[pos])
    #for i in results['validation']:
    #    vvals.append(i[pos])
    for i in results['test']:
        tsvals.append(i[pos])
    print("Initial Test NDCG@",indexer[pos],":",np.round(tsvals[0],4)," Final Test NDCG@",indexer[pos],":",np.round(tsvals[-1],4))

    #axs.plot(tvals, color='green', label='training')
    #axs.plot(vvals, marker='o', markersize='5', color='orange', linestyle='dashed', label='validation')
    axs.plot(tsvals,  color='red', label='testing')
    axs.grid()
    axs.legend()
    

ap=argparse.ArgumentParser()
ap.add_argument("-d","--datadir",required=True,help="Path to data file to visualize")
args=vars(ap.parse_args())
path=str(args["datadir"])


with open(path,'rb') as handle:
    results=pickle.load(handle)

fig = plt.figure(figsize=(26, 17))
grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.6, figure=fig)

ax1 = fig.add_subplot(grid[0, 0])
ax2 = fig.add_subplot(grid[0, 1])
ax3 = fig.add_subplot(grid[1, 0])
ax4 = fig.add_subplot(grid[1, 1])


#Trends for NDCG@1,3,5,10 and NDCG@K

ax1.set_title("NDCG@1")
make_plot(results, axs=ax1, pos=1)

ax2.set_title("NDCG@3")
make_plot(results, axs=ax2, pos=2)

ax3.set_title("NDCG@5")
make_plot(results, axs=ax3, pos=3)

ax4.set_title("NDCG@10")
make_plot(results, axs=ax4, pos=4)

plt.show()