import pickle 
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def make_plot(results, axs):
    """Takes the results dictionary, k value as input and plots a graph of train NDCG@k,validation NDCG@k and test NDCG@k"""
    tsvals= results
    axs.plot(tsvals,  color='blue', label='training')
    axs.grid()
    axs.legend()

def visualize(train_results, outfile):
    fig = plt.figure(figsize=(17, 17))
    grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.6, figure=fig)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])

    ax1.set_title("NDCG@1")
    make_plot([x[0][0] for x in train_results], axs=ax1)

    ax2.set_title("NDCG@3")
    make_plot([x[0][2] for x in train_results], axs=ax2)

    ax3.set_title("NDCG@5")
    make_plot([x[0][4] for x in train_results], axs=ax3)

    ax4.set_title("NDCG@10")
    make_plot([x[0][9] for x in train_results], axs=ax4)

    plt.savefig(outfile + ".png")
    plt.show()