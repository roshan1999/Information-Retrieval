### Runs PPO on MDPRank setting for selected hyperparameters using optimal action (sampling best action) on 70-30 Train Test Split

---


To run the algorithm, enter the following code:<br>

`$ python main.py -d data_directory -nf num_features -i num_iterations -g gamma -alpha learningrate -beta learningrate -hnodes nodes_hiddenlayer -steps episode_length 
-batch_size mini_batch_size -epochs num_epochs -clip policy_clip -lambda smoothing_factor`<br>

<br>Default hyperparameters: 
---
1. Data(-d): `Required`
2. Number of features(-nf): `Required`
3. Number of iterations(-i): `Required`
4. Gamma(-g): `1`
6. Learning rate actor(-alpha): `0.00001`
7. Learning rate critic (-beta): `0.00002`
8. Hidden layer Nodes(-hnodes): `46`
9. Episode Length(-steps): `50`
10. Mini-batch-size(-batch_size): `5`
11. Epochs(-epochs): `3`
12. Policy clip (-clip): `0.1`
13. Smoothing factor(-lambda): `0.95`


<br>Example: 
---
1. Running on default hyperparameters for 100 iterations (MQ2008): <br> `$ python main.py -d ../data/MQ2008/all_0,1,2 -i 100 -nf 46`
2. Running on default hyperparameters for 100 iterations (MQ2007): <br> `$ python main.py -d ../data/MQ2007/all_0,1,2 -i 100 -nf 46`
3. Running with given hyperparameters: <br> `$ python main.py -d ../data/MQ2008 -i 100 -nf 46 -g 1 -alpha 0.00001 -beta 0.00002 -hnodes 46 -steps 50 -batch_size 5 -epochs 3 -clip 0.1 -lambda 0.95`


---
<br>PPO Algorithm:

<img src="https://user-images.githubusercontent.com/51087175/121606087-2837f780-ca6b-11eb-8560-e31ad96f05b3.png" width="700" height="350">

<br>PPORank Algorithm:

<img src="https://user-images.githubusercontent.com/51087175/125517857-818f339f-fc93-4d2a-9b07-40b91bd88ddd.png" width="700" height="600">
<img src="https://user-images.githubusercontent.com/51087175/125517891-454181a0-ee82-4449-8768-59f4e1dffca2.png" width="700" height="400">

---
