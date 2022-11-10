### Runs Linear Incremental Actor Critic on MDPRank setting using given hyperparameters

---


To run the algorithm, enter the following code:<br>
`$ python main_incremental.py -d data_directory -nf no_features -e no_epochs -g gamma -lr_a actor_lr -lr_c critic_lr -nstep length_episode -trainsize training_set_proportion -seed seed  `<br>


<br>Example: 
---
1. Running on given hyperparameters: <br> `$ python main_incremental.py -d ./Data/OHSUMED_All -nf 45 -lr_a 0.0001 -lr_c 0.0002 -g 1 -e 100 -nstep 30 -seed 7 -trainsize 70`


<br>
This runs Linear Incremental Actor Critic for given hyperparameters and saves all results,graphs,models in the Folder, Results/Dataset.


---

<br>IncrementalRank Algorithm:

<img src="https://user-images.githubusercontent.com/51087175/125698682-661349fa-a85d-4f0c-b279-eaa10f3b7152.png" width="540" height="450">
