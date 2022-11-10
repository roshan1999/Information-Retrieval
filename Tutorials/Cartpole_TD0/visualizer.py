import matplotlib.pyplot as plt
import pickle
import pandas as pd

f = open('./Results/scores_record_2500_0.99_0.0001_0.0005', 'rb')
t = pickle.load(f)
smoothed_rewards = pd.Series.rolling(pd.Series(t), 50).mean()
smoothed_rewards = [elem for elem in smoothed_rewards]
plt.plot(smoothed_rewards)
plt.xlabel('Number of episodes')
plt.ylabel('Total Episodic Reward')
plt.show()
