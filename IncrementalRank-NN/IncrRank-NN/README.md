### Runs Neural Network implementation of Incremental Actor Critic on MDPRank setting using the given hyperparameters

---
Link to combined Dataset for MQ2007 and MQ2008: https://drive.google.com/drive/folders/18U4rHifbOWgkuQIj_ECoDqige1ypELpQ?usp=sharing

To run the algorithm, enter the following code:<br>
`$ python main.py -d data_directory -nf no_features -e no_epochs -g gamma -alpha actor_lr -beta critic_lr -length length_episode`<br>

<br>Default hyperparameters: 
---
1. Data(-d): `Required`
2. Number of features(-nf): `Required`
3. Number of epochs (-e): `Required`
4. Gamma(-g): `1`
5. Learning rate actor(-alpha): `1e-05`
6. Learning rate critic (-beta): `2e-05`
7. Episode Length(-length): `20`

<br>Example: 
---
1. Running on default hyperparameters for MQ2008: `$ python main.py -d ./data/MQ2008/all_0,1,2 -nf 46 -e 100`
2. Running on default hyperparameters for MQ2007: `$ python main.py -d ./data/MQ2007/all_0,1,2 -nf 46 -e 100`
3. Running on given hyperparameters for MQ2008: <br> `$ python main.py -d ./data/MQ2008/all_0,1,2 -nf 46 -alpha 1e-05 -beta 2e-05 -g 1 -e 100 -length 20`
4. Running on given hyperparameters for MQ2007: <br> `$ python main.py -d ./data/MQ2007/all_0,1,2 -nf 46 -alpha 1e-05 -beta 2e-05 -g 1 -e 100 -length 20`

#### Note:
- This runs Non-Linear Incremental Actor Critic for given hyperparameters and saves all results,graphs,models in the Folder, Results/Dataset.
- This version has 2 hidden layers for the actor (TanH activation) and critic (RELU activation) with 64 and 32 nodes 

<br>IncrementalRank Algorithm:

<img src="https://user-images.githubusercontent.com/51087175/125698682-661349fa-a85d-4f0c-b279-eaa10f3b7152.png" width="540" height="450">
---


