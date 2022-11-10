### Runs Linear Pairwise Incremental Actor Critic on MDPRank setting using given hyperparameters

---


To run the algorithm, enter the following code:<br>
`$ python main_incremental.py -d data_directory -nf no_features -e no_epochs -g gamma -lr_a actor_lr -lr_c critic_lr -nstep length_episode -trainsize training_set_proportion -seed seed  `<br>


<br>Example: 
---
1. Running on given hyperparameters: <br> `$ python main_incremental.py -d ./Data/MQ2007_All_DIC_0,1,2 -nf 46 -lr_a 0.0001 -lr_c 0.0002 -g 1 -e 30 -nstep 20 -seed 3 -trainsize 70`


<br>
This runs Linear Pairwise Incremental Actor Critic for given hyperparameters and saves all results,graphs,models in the Folder, Results/Dataset.


---

<br>IncrementalRank-Pairwise Algorithm:

<img src="https://user-images.githubusercontent.com/51087175/125940009-010624ba-35c6-4741-a7cb-09c9408c7280.png" width="560" height="460">

