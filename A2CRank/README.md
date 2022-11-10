### Runs Linear A2C on MDPRank setting using given hyperparameters

---


To run the algorithm, enter the following code:<br>
`$ python main_A2C.py -d data_directory -nf no_features -e no_epochs -g gamma -lr_actor actor_lr -lr_critic critic_lr -episode_length episode_length -seed seed `<br>


<br>Example: 
---
1. Running on given hyperparameters: <br> `$ python main_A2C.py -d ./Data/MQ2008_All_DIC_0,1,2 -nf 46 -lr_actor 0.0001 -lr_critic 0.0002 -g 1 -e 50 -episode_length 30 -seed 3 `
2. Running on given hyperparameters: <br> `$ python main_A2C.py -d ./Data/MQ2007_All_DIC_0,1,2 -nf 46 -lr_actor 0.0001 -lr_critic 0.0002 -g 1 -e 50 -episode_length 20 -seed 3 `


<br>
This runs Linear A2C for the given hyperparameters and saves all results,graphs,models in the Folder, Results/Dataset.

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

<img src="https://user-images.githubusercontent.com/51087175/121325910-a68f7f00-c92f-11eb-8e95-dd72e7a26a74.jpeg" width="400" height="500">

---

