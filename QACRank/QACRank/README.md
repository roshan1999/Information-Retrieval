Readme
### Runs Linear Q Actor Critic Rank with compatible features on MDPRank setting using given hyperparameters

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
1. Running step updates on given hyperparameters: <br> `$ python main_qac_step.py -d ./Data/MQ2007_All_DIC_0,1,2 -nf 46 -lr_a 0.0001 -lr_c 0.0002 -g 1 -e 50 -seed 7 -trainsize 70 `
2. Running episode updates on given hyperparameters: <br> `$ python main_qac_episodic.py -d ./Data/MQ2007_All_DIC_0,1,2 -nf 45 -lr_a 0.0001 -lr_c 0.0002 -g 1 -e 50 -seed 7 -trainsize 70 `
3. Running epoch updates on given hyperparameters: <br> `$ python main_qac_epoch.py -d ./Data/OHSUMED_All -nf 45 -lr_a 0.005 -lr_c 0.009 -g 1 -e 50 -seed 7 -trainsize 70 `


<br>
This runs Linear QACRank for given hyperparameters and saves all results,graphs,models in the Folder, Results/Dataset.

---

<br>Visualization:


Plot avgstepreward vs epoch graph : <br> `$ python visualize_return.py -d path_to_datafile`<br>

For example: `$ python visualize_return.py -d ./Results/MQ2007/MQ2007_e50_lr_a,c,0.0001,0.0002_g1.0_allSplit_ustep_irandn_seed7_data` <br>

<br>

Plot ndcg vs epoch graph : <br> `$ python visualizer.py -d path_to_datafile`<br>

For example: `$ python visualizer.py -d ./Results/MQ2007/MQ2007_e50_lr_a,c,0.0001,0.0002_g1.0_allSplit_ustep_irandn_seed7_data` 

---
<br>QACRank-step updates Algorithm:

<img src="https://user-images.githubusercontent.com/51087175/125998809-54a1af8a-683a-47ae-8988-d6696ac07368.png" width="600" height="400">

<br>Loss Functions:

<img src="https://user-images.githubusercontent.com/51087175/126051846-e2869381-d2c7-46f6-95bc-106d64375333.png" width="400" height="200">

---
