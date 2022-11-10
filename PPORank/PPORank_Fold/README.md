### Runs PPO on MDPRank setting for selected hyperparameters for each fold

---


To run the algorithm, enter the following code:<br>
`$ python main.py -d data_directory -f fold_number -nf num_features -i num_iterations -g gamma -alpha learningrate -hnodes nodes_hiddenlayer -steps episode_length 
-batch_size mini_batch_size -epochs num_epochs -clip policy_clip -lambda smoothing_factor -c1 value_fun_coef -c2 entropy`<br>

<br>Default hyperparameters (Also the optimal for MQ2008): 
---
1. Data(-d): `Required`
2. Fold(-f): `Required`
3. Number of features(-nf): `Required`
4. Number of iterations(-i): `Required`
5. Gamma(-g): `1`
6. Actor Learning rate(-alpha): `0.0001`
7. Critic Learning rate(-beta): `0.0002`
8. Hidden layer Nodes(-hnodes): `45`
9. Episode Length(-steps): `20`
10. Mini-batch-size(-batch_size): `5`
11. Epochs(-epochs): `1`
12. Policy clip (-clip): `0.1`
13. Smoothing factor(-lambda): `1.0`
14. Value_function coeficient(-c1): `0.5`
15. Entropy Coeficient(-c2): `0.01`  


<br>Example: 
---
1. Running on optimal/default hyperparameters for fold 1 and 50 iterations (MQ2008): <br> `$ python main.py -d ../data/MQ2008 -f 1 -i 50 -nf 46`
2. Running on default hyperparameters for fold 1 and 50 iterations (MQ2007): <br> `$ python main.py -d ../data/MQ2007 -f 1 -i 50 -nf 46`
3. Running on default hyperparameters for fold 1 and 50 iterations (MSLR-WEB10K): <br> `$ python main.py -d ../data/MSLR-WEB10K -f 1 -i 50 -nf 136`
3. Running with given hyperparameters: <br> `$ python main.py -d ../data/MQ2008 -i 50 -nf 46 -f 1 -g 0.98 -alpha 0.005 -beta 0.006 -hnodes 45 -steps 30 -batch_size 5 -epochs 4 -clip 0.3 -lambda 0.99 -c1 0.5 -c2 0.01`


<br>
This runs PPORank for given hyperparameters and saves all results,graphs,models in the Folder, Results.

---


---
<br>PPO Algorithm:

<img src="https://user-images.githubusercontent.com/51087175/121606087-2837f780-ca6b-11eb-8560-e31ad96f05b3.png" width="700" height="350">

<br>PPORank Algorithm:

<img src="https://user-images.githubusercontent.com/51087175/121606091-2a01bb00-ca6b-11eb-84d5-e6cd3a28d905.jpeg" width="400" height="600">

---


