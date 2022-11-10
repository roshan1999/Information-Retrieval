### Runs Neural Network implementation of Advantage Actor Critic on MDPRank setting using given hyperparameters

---
Link to combined Dataset for MQ2007 and MQ2008: https://drive.google.com/drive/folders/18U4rHifbOWgkuQIj_ECoDqige1ypELpQ?usp=sharing

To run the algorithm, enter the following code:<br>
`$ python main.py -d data_directory -nf no_features -e no_epochs -g gamma -alpha actor_lr -beta critic_lr -s length_episode -layer hidden_layer_nodes`<br>

<br>Default hyperparameters: 
---
1. Data(-d): `Required`
2. Number of features(-nf): `Required`
3. Number of epochs (-e): `Required`
4. Gamma(-g): `1`
5. Learning rate actor(-alpha): `0.0001`
6. Learning rate critic (-beta): `0.0002`
7. Hidden layer Nodes(-layer): `46`

<br>Example: 
---
1. Running on default hyperparameters for MQ2008 (70-30) split: `$ python main.py -d ../data/MQ2008/all_0,1,2 -nf 46 -e 200`
2. Running on default hyperparameters for MQ2007 (70-30): `$ python main.py -d ../data/MQ2007/all_0,1,2 -nf 46 -e 200`
3. Running on given hyperparameters for MQ2008 (70-30): <br> `$ python main.py -d ../data/MQ2008/all_0,1,2 -nf 46 -alpha 0.0001 -beta 0.0002 -g 1 -e 100`
4. Running on given hyperparameters for MQ2007 (70-30): <br> `$ python main.py -d ../data/MQ2007/all_0,1,2 -nf 46 -alpha 0.0001 -beta 0.0002 -g 1 -e 100`

#### Note:
- This runs Non-Linear A2C for given hyperparameters and saves all results,graphs,models in the Folder, Results/Dataset.
- The results are generated for a single hidden layer on the MQ2008 and MQ2007 dataset. This code reproduces the results

---
<br>References for A2C:
---
1. https://julien-vitay.net/deeprl/ActorCritic.html
2. https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
3. https://github.com/ashish-interns/DeepRL/tree/master/Tutorials/Actor-Critic
---
<br>A2CRank Algorithm:

<img src="https://user-images.githubusercontent.com/51087175/125530931-5e36bd4b-ba62-4fe4-8842-3c1a101e6956.png" width="535" height="400">

<br>A2CRank Loss Functions:

<img src="https://user-images.githubusercontent.com/51087175/121325910-a68f7f00-c92f-11eb-8e95-dd72e7a26a74.jpeg" width="324" height="400">

---


