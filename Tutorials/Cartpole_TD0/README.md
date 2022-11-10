### Runs CartPole_v0 using incremental actor critic for selected hyperparameters

---


To run the algorithm, enter the following code:<br>
`$ python main.py -e no_epochs -g gamma -alpha actor_lr -beta critic_lr -layer1 nodes_hiddenlayer1 -layer2 nodes_hiddenlayer2 -interval disp_interval `<br>

<br>Default hyperparameters: 
---
1. Episodes: `2500`
2. Actor Learning rate: `0.0004`
3. Critic Learning rate: `0.0005`
4. Gamma: `0.99`
5. hidden-layer1 nodes: `32`
6. hidden-layer2 nodes: `32`
7. interval: `20`


<br>Example: 
---
1. Running on default hyperparameters: <br> `$ python main.py `
2. Running with given hyperparameters: <br> `$ python main.py -e 1500 -g 1 -alpha 1e-4 -beta 5e-4 -layer1 20 -layer2 20 -interval 15 `


<br>
This runs incremental actor critic for given hyperparameters and saves all results,graphs,models in the Folder, Results.

---
