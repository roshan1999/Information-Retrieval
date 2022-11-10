# Dynamic Ranking Simulator
### Files to download
---
[MQ2007 Combined dataset](https://drive.google.com/drive/folders/1BVaNzL1hMeDeajJU6oXEdsr5VJDi0llE?usp=sharing)


To run the algorithm, enter the following code:<br>
`$ python ./visual.py -fname filename -M init_docs -extra add_docs -d decay_value -nf no_features -g gamma -alpha hyperparam1 -beta hyperparam2 -lr learning_rate -window window_size -size corpus_size`<br>

<br>Default hyperparameters: 
---
1. FileName of dataset to load (-fname) : `Required`
2. Initial M documents (-M) : `Required`
3. Extra documents to add at each time step (-extra) : `Required`
4. Decay(-d): `Required`
5. Number of features(-nf): `Required`
6. Gamma(-g): `1`
7. Learning rate(-lr): `0.001`
8. Hyper Parameter SS retrieval(-alpha): `0.1`
8. Hyper Parameter RS retrieval(-beta): `0.01`
9. Sliding Window Size(-window): `20000`

Example : <br>
<br>
For MQ2007 dataset: `$ python ./visual.py -fname ./MQ2007/combined -M 10000 -extra 1000 -nf 46 -d 0.01`<br>

For MQ2008 dataset: `$ python ./visual.py -fname ./MQ2008/combined -M 10000 -extra 1000 -nf 46 -d 0.01`<br>

Note: This runs the algorithm for initialized 10000 documents with extra 1000 documents at each time step added. The decay is 0.01

