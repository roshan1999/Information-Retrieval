Readme
### Runs Linear Pairwise Q Actor Critic Rank on MDPRank setting using given hyperparameters

---


To run the algorithm enter the following code:<br>
`$ python main_qac_pairwise.py -d data_directory -nf no_features -e no_epochs -g gamma -lr_a actor_lr -lr_c critic_lr -trainsize training_set_proportion -seed seed  `<br>

<br>Example: 
---
1. Running step updates on given hyperparameters: <br> `$ python main_qac_pairwise.py -d ./Data/MQ2008_All_DIC_0,1,2 -nf 46 -lr_a 0.0001 -lr_c 0.0002 -g 1 -e 50 -seed 7 `
2. Running step updates on given hyperparameters: <br> `$ python main_qac_pairwise.py -d ./Data/MQ2007_All_DIC_0,1,2 -nf 46 -lr_a 0.0001 -lr_c 0.0002 -g 1 -e 50 -seed 3 `


<br>
This runs Linear QACRank Pairwise for given hyperparameters and saves all results,graphs,models in the Folder, Results/Dataset.

---

<br>Visualization:


Plot avgstepreward vs epoch graph : <br> `$ python visualize_return.py -d path_to_datafile`<br>

For example: `$ python visualize_return.py -d ./Results/MQ2007/MQ2007_e50_lr_a,c,0.0001,0.0002_g1.0_allSplit_pairwise_ustep_irandn_seed3_data` <br>

<br>

Plot ndcg vs epoch graph : <br> `$ python visualizer.py -d path_to_datafile`<br>

For example: `$ python visualizer.py -d ./Results/MQ2007/MQ2007_e50_lr_a,c,0.0001,0.0002_g1.0_allSplit_pairwise_ustep_irandn_seed3_data`  

---
<br>QACRank-Pairwise Algorithm:

<img src="https://user-images.githubusercontent.com/51087175/126392895-923f29f3-810b-4c46-9e6c-5271582f5225.png" width="570" height="460">

---
