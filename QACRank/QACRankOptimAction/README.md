Readme
### Runs Linear Q Actor Critic Rank optimal action on MDPRank setting using given hyperparameters

---


To run the algorithm for step updates, enter the following code:<br>
`$ python main_qac_step.py -d data_directory -nf no_features -e no_epochs -g gamma -lr_a actor_lr -lr_c critic_lr -trainsize training_set_proportion -seed seed  `<br>
<br>
To run the algorithm for episodic updates, enter the following code:<br>
`$ python main_qac_episodic.py -d data_directory -nf no_features -e no_epochs -g gamma -lr_a actor_lr -lr_c critic_lr -trainsize training_set_proportion -seed seed `<br>
<br>
To run the algorithm for epoch updates, enter the following code:<br>
`$ python main_qac_epoch.py -d data_directory -nf no_features -e no_epochs -g gamma -lr_a actor_lr -lr_c critic_lr -trainsize training_set_proportion -seed seed  `<br>


<br>Example: 
---
1. Running step updates on given hyperparameters: <br> `$ python main_qac_step.py -d ./Data/MQ2008_All_DIC_0,1,2 -nf 46 -lr_a 0.0001 -lr_c 0.0002 -g 1 -e 100 -seed 7 -trainsize 70 `
2. Running step updates on given hyperparameters: <br> `$ python main_qac_step.py -d ./Data/MQ2007_All_DIC_0,1,2 -nf 46 -lr_a 0.0001 -lr_c 0.0002 -g 1 -e 100 -seed 7 -trainsize 70 `


<br>
This runs Linear QACRank optimal action for given hyperparameters and saves all results,graphs,models in the Folder, Results/Dataset.

---

<br>Visualization:


Plot test ndcg vs epoch graph : <br> `$ python visualizer.py -d path_to_datafile`<br>

For example: `$ python visualizer.py -d ./Results/MQ2007/MQ2007_e100_lr_a,c,0.0001,0.0002_g1.0_allSplit_optimaction_ustep_irandn_seed7_data` 

---
<br>QACRank-optimaction-step updates Algorithm:

<img src="https://user-images.githubusercontent.com/51087175/126051669-6d13528a-abe0-4100-8a92-21aae799329a.png" width="550" height="400">


<br>Loss Functions:

<img src="https://user-images.githubusercontent.com/51087175/126051846-e2869381-d2c7-46f6-95bc-106d64375333.png" width="400" height="200">

---
