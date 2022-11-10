import pickle
import matplotlib.pyplot as plt


def make_plot(results, axs, pos):
    """Takes the results dictionary, k value as input and plots a graph of train NDCG@k,validation NDCG@k and test NDCG@k"""
    indexer = {0: 0, 1: 1, 3: 2, 5: 3, 10: 4}
    tvals = []
    vvals = []
    testval = []
    for i in results['Training']:
        tvals.append(i[pos])
    for i in results['Validation']:
        vvals.append(i[pos])
    for i in results['Test']:
    	testval.append(i[pos])	

    axs.plot(tvals, marker='*', markersize='7', color='green', linestyle='dashed', label='Training')
    axs.plot(vvals, marker='o', markersize='5', color='orange', linestyle='dashed', label='Validation')
    axs.plot(testval, marker=".", markersize='7', color='red', label='Testing')
    axs.legend()

fold = int(input("Enter the fold number to visualize: "))

with open('fold_' + str(fold) + '_data', 'rb') as handle:
    results = pickle.load(handle)

fig = plt.figure(figsize=(8, 12))
grid = plt.GridSpec(3, 2, wspace=0.4, hspace=0.6, figure=fig)
ax0 = fig.add_subplot(grid[0, :])
ax1 = fig.add_subplot(grid[1, 0])
ax2 = fig.add_subplot(grid[1, 1])
ax3 = fig.add_subplot(grid[2, 0])
ax4 = fig.add_subplot(grid[2, 1])

ax0.set_title("NDCG@K")
make_plot(results, axs=ax0, pos=0)

ax1.set_title("NDCG@1")
make_plot(results, axs=ax1, pos=1)

ax2.set_title("NDCG@3")
make_plot(results, axs=ax2, pos=2)

ax3.set_title("NDCG@5")
make_plot(results, axs=ax3, pos=3)

ax4.set_title("NDCG@10")
make_plot(results, axs=ax4, pos=4)

plt.show()
