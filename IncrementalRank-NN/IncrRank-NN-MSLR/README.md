### Runs Neural Network implementation of Incremental Actor Critic on MDPRank setting using the given hyperparameters

---
Link to combined Dataset for MSLRWEB10k: https://drive.google.com/drive/u/0/folders/130ig_I1cUY4PdyAg0BPBS3XuBqn-Kccs

To run the algorithm, enter the following code:<br>
`$ python main.py -d data_directory -nf no_features -e no_epochs -g gamma -alpha actor_lr -beta critic_lr -length length_episode`<br>

<br>Default hyperparameters: 
---
1. Data(-d): `Required`
2. Number of features(-nf): `Required`
3. Number of epochs (-e): `Required`
4. Gamma(-g): `1`
5. Learning rate actor(-alpha): `1e-06`
6. Learning rate critic (-beta): `2e-06`
7. Episode Length(-length): `20`

<br>Example: 
---
1. Running on default hyperparameters: `$ python main.py -d ./MSLR-WEB10K/MSLR_combined -e 50 -nf 136`

#### Note:
- This runs Non-Linear Incremental Actor Critic for given hyperparameters and saves all results,graphs,models in the Folder, Results/Dataset. 

<br>IncrementalRank Algorithm:

<img src="https://user-images.githubusercontent.com/51087175/125698682-661349fa-a85d-4f0c-b279-eaa10f3b7152.png" width="540" height="450">
---


