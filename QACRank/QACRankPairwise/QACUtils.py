import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

def get_reward(a,t,data,q):
    """Given the action taken at time t returns the reward"""
    label=data[q][a][1]
    if(t==0):
        return 2**label-1
    else:
        return float((2**label-1)/(math.log(t+1,2)))

def sample_action(theta,s,data,q):
    """Given the state and policy returns the action taken"""

    docidsleft=s[1]
    X=[]#feature matrix, features of all docs in the current state.
    for docid in docidsleft:
        X.append(data[q][docid][0])
    X=np.array(X)#(ndocs,nfeatures)
    scores=np.matmul(theta.T,X.T).T # (ndocs,1)
    scores=np.round(scores,5)
    probs=np.exp(scores)/np.sum(np.exp(scores))#softmax
    return np.random.choice(docidsleft,p=probs.reshape(len(probs)))

def sample_best_action(theta,s,data,q):
    """Given the state returns the best possible action according to trainset"""
    docidsleft=s[1]
    labels=[]
    for docid in docidsleft:
        labels.append(data[q][docid][1])
    return docidsleft[np.argmax(np.array(labels))]

def compute_scorefn(s,a,theta,data,q):
    """Takes the current state,action,theta and computes the scorefunction """
    #selected features- sum(features*prob(selectfeatures))
    docidsleft=s[1]
    X=[]#feature matrix, features of all docs in the current state.
    for docid in docidsleft:
        X.append(data[q][docid][0])
    X=np.array(X)#(ndocs,nfeatures)
    scores=np.matmul(theta.T,X.T).T # (ndocs,1)
    scores=np.round(scores,5)
    probs=np.exp(scores)/np.sum(np.exp(scores))#softmax
    return np.round(data[q][a][0]-np.sum(X*probs,axis=0),5).reshape(len(theta),1)

def compute_qvalue(w,psi_s_a):
    """Given the critic weights and psi feature vector computes the qvalue"""
    return float(np.dot(w.T,psi_s_a))

def add_dim(grad_s_a,t):
    """Adds a dimension t at the start and returns the vector"""
    grad_s_a=grad_s_a.reshape(len(grad_s_a))
    psi_s_a=list(grad_s_a)
    psi_s_a.insert(0,t)
    psi_s_a=np.array(psi_s_a).reshape(len(grad_s_a)+1,1)
    return psi_s_a


def view_save_plot(results,outfile):
    """Takes the results dictionary and plots avg step reward graph"""
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
    plt.title("QACRank(Linear)")
    plt.plot(lst_step, markersize='5', color='blue', label='Avg Step Reward')
    plt.plot(smooth_returns, linewidth=3,markersize='5', color='green', label='Smooth Avg Step Reward')
    plt.ylabel("Avg Step Reward")
    plt.xlabel("Epochs")
    plt.grid()
    plt.legend()
    plt.savefig(outfile+"_avgstepreward.png")
    plt.show()

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

    axs.plot(tvals, color='green', label='training')
    #axs.plot(vvals, marker='o', markersize='5', color='orange', linestyle='dashed', label='validation')
    axs.plot(tsvals,  color='red', label='testing')
    axs.grid()
    axs.legend()


def view_save_plot_ndcg(results,outfile):
    """Plots the ndcg 1,3,5,10 graphs and saves it in outfile"""
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

    plt.savefig(outfile+"_ndcg.png")
    plt.show()
