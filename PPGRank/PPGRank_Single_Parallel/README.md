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
|-np| --num_process | No; Default = 4 | Number of process to run in parallel |


<br>For example: 
---
1. Running on OHSUMED dataset: <br> `python main.py --data ./data/OHSUMED/ -e 50 -f 1 -nf 45 `
2. Running on MQ2008 dataset: <br> `python main.py --data ./data/MQ2008/ -e 50 -f 1 -nf 46 -g 1`
4. Running on MQ2007 dataset: <br> `python main.py -d ./data/MQ2007/ -e 75 -f 1 -nf 136 -g 1`
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

