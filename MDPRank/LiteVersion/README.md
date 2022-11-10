### Runs MDPRank for a selected fold and hyperparameters

---


To run the algorithm, enter the following code:<br>
`$ python main.py -d datapath -nf number_features -e no_epochs -f fold -lr learning_rate `<br>

<br>For example: 
---
1. Running on MQ2008 dataset: <br> `$ python main.py -d ./MDPRank/Data/MQ2008 -e 100 -nf 46 -f 1 -lr 0.05`
2. Running on OHSUMED 45 dataset: <br> `$ python main.py -d ./MDPRank/Data/OHSUMED/QueryLevelNorm -e 100 -nf 45 -f 1 -lr 0.05`
3. Running on MQ2007 dataset: <br> `$ python main.py -d ./MDPRank/Data/MQ2007 -e 50 -nf 46 -f 1 -lr 0.05`


<br>
This runs MDPRank for a particular fold and saves the results in fold_foldno_weight and fold_foldno_data.

---

<br>Further Analysis:


- Can visualize by running: <br> `$ python visualizer.py -f foldnumber`<br>
For example: 

- Visualizing 1st fold <br> `$ python visualizer.py -f 1` 

> Note: fold_foldno_data files are in the same directory as visualizer.py

<br>

- To see the average and max over all the folds, run:<br> `$ python averager.py`

> Note: fold_foldno_data files are in the same directory as averager.py

---


---
<br> 

## Observations


### Results

| Dataset | Epochs | Learning Rate | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| MQ2008 | 150 | 0.05 | 0.3741 | 0.4099 | 0.4569 | 0.2230 |
| MQ2007 | 50 | 0.05 | 0.3954 | 0.3983 | 0.4069 | 0.435 |


### Paper Results

| Dataset | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| MQ2008 | 0.3827 | 0.4420 | 0.4881 | 0.2327 |
| MQ2007 | 0.4061 | 0.4101 | 0.4171 | 0.4416 |
