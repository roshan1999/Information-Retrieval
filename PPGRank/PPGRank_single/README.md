# PPGRank
---
### Implementation of PPGRank without Cross-Validation

- Arguments are to be parsed <br>
- To see all arguments run: `python main.py --help`
<br>
- Different arguments both default and required are as follows:<br>
<br>

|Option | argument param | Required? | Description |
|---|---|---|---|
|-d | --data | Yes | relative or absolute directory of the data where the folds exist |
|-e | --epoch | Yes | number of epochs to run for|
|-f | --fold | Yes | fold number to run|
|-nf| --features_no | Yes | number of features in the dataset |
|-g | --gamma| No; Default = 1 | hyperparameter gamma value |
|-alpha| --learning_rate | No; Default = 0.005 | Learning rate of the agent |
|-dp| --display | No; Default = True | Displays each query NDCG while calculating, else prints after entire query set calculated |


<br>For example: 
---
1. Running on OHSUMED 45 dataset: <br> `python main.py --data ./data/OHSUMED/data -e 30 -f 1 -nf 45 -g 1 -alpha 0.005 -dp True`
2. Running on MQ2008 dataset: <br> `python main.py --data ./data/MQ2008/data -e 30 -f 1 -nf 46 -g 1 -alpha 0.005 -dp True`
3. Running on MSLRWEB10K dataset: <br> `python main.py -d ./data/MSLR-WEB10K -e 75 -f 1 -nf 136 -g 1 -alpha 0.005 -dp True`
---

<br>Extra:
---
- MSLRWEB10k Dataset download: https://1drv.ms/u/s!AtsMfWUz5l8nbOIoJ6Ks0bEMp78
- Running these will store the results and the weights with the respective folds in the files `fold_{f}_data` and `fold_{f}_weights` in the current directory.
- Once the main python file has completed running, for visualization use: <br> `python visualizer.py`
> Note: The fold to visualize is asked as user input. **Make sure the fold exists in the same directory as the visualizer.py file**

- Once all folds are run, to know the average and maximum across all folds, run:<br>
`python pickle_reader.py -d ./` 

  - Use -d option to mention the location of all the folds saved for ex: `python .\pickle_reader.py -d ./analysis`

## Observation:
---

### Average scores

| Dataset | alpha | gamma | epochs | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| OHSUMED | 0.005 | 1 | 30 | 0.5545 | 0.5302 | 0.5077 | 0.4920 |
| MQ2008 | 0.005 | 1 | 30 | 0.3882 | 0.4309 | 0.4694 | 0.5138 |

### Paper's Scores

| Dataset | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| OHSUMED | 0.5771 | 0.5218 | 0.4911 | 0.4664 |
| MQ2008 | 0.3877 | 0.4511 | 0.4910 | 0.2455 |

## Analysis

- [Click here for visualization of each fold - MQ2008](https://github.com/ashish-interns/DeepRL/tree/master/PPGRank/PPGRank_single/analysis/MQ2008)
- [Click here for visualization of each fold - OHSUMED](https://github.com/ashish-interns/DeepRL/tree/master/PPGRank/PPGRank_single/analysis/OHSUMED)


