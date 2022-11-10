import pickle 
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def view_save_plot(results,outfile):
    """Plot the avg step reward graph"""
    lst_returns=[]
    lst_avgreturns=[]
    lst_step=[]
    for i in results['training']:
        lst_returns.append(i[-2])
        lst_avgreturns.append(i[-1])
        lst_step.append(i[-3])
    lst_avgreturns=lst_avgreturns[1:]
    lst_returns=lst_returns[1:]
    lst_step=lst_step[1:]

    smooth_returns = pd.Series.rolling(pd.Series(lst_step), 10).mean()
    plt.title("QACRank(Linear)")
    plt.plot(lst_step, markersize='5', color='blue', label='Avg Step Reward')
    plt.plot(smooth_returns, linewidth=3,markersize='5', color='green', label='Smooth Avg Step Reward')
    plt.ylabel("Avg Step Reward")
    plt.xlabel("Epochs")
    plt.grid()
    plt.legend()
    #plt.savefig(outfile+"_step.png")
    plt.show()


ap=argparse.ArgumentParser()
ap.add_argument("-d","--datadir",required=True,help="Path to data file to visualize")
args=vars(ap.parse_args())
path=str(args["datadir"])


with open(path,'rb') as handle:
	results=pickle.load(handle)


view_save_plot(results, "outfile.png")
