### Runs Linear Incremental Actor Critic optimal action on MDPRank setting using given hyperparameters

---


To run the algorithm, enter the following code:<br>
`$ python main_incremental.py -d data_directory -nf no_features -e no_epochs -g gamma -lr_a actor_lr -lr_c critic_lr -nstep length_episode -trainsize training_set_proportion -seed seed  `<br>


<br>Example: 
---
1. Running on given hyperparameters: <br> `$ python main_incremental.py -d ./MQ2008_All_DIC_0,1,2 -e 100 -nf 46 -lr_a 0.0001 -lr_c 0.0002 -g 1 -nstep 50 -seed 3 -trainsize 70 `
2. Running on given hyperparameters: <br> `$ python main_incremental.py -d ./MQ2007_All_DIC_0,1,2 -e 100 -nf 46 -lr_a 0.0001 -lr_c 0.0002 -g 1 -nstep 50 -seed 3 -trainsize 70 `


<br>
This runs Linear Incremental Actor Critic optimal action for given hyperparameters and saves all results,graphs,models in the Folder, Results/Dataset.


---

<br>IncrementalRank-optimaction Algorithm:

<img src="https://user-images.githubusercontent.com/51087175/125865002-007b41fe-2e1a-47d9-8104-ee72bc4bd0da.png" width="540" height="450">

