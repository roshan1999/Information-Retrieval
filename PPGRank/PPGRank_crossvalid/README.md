# PPGRank
---
### Implementation of PPGRank with cross validation

- Arguments are to be parsed <br>
- To see all arguments run: `python main.py --help`
<br>
- Different arguments both default and required are as follows:<br>
<br>

|Option | argument param | Required? | Description |
|---|---|---|---|
|-d | --data | Yes | relative or absolute directory of the data where the folds exist |
|-e | --epoch | Yes | number of epochs to run for|
|-nf| --features_no | Yes | number of features in the dataset |
|-g | --gamma| No; Default = 1 | hyperparameter gamma value |
|-alpha| --learning_rate | No; Default = 0.005 | Learning rate of the agent |
|-dp| --display | No; Default = True | Displays each query NDCG while calculating, else prints after entire query set calculated |


<br>For example: 
---
1. Running on OHSUMED 45 dataset: <br> `python main.py --data ./data/OHSUMED/data -e 30 -nf 45 -g 1 -alpha 0.005 -dp True`
2. Running on MQ2008 dataset: <br> `python main.py --data ./data/MQ2008/data -e 30 -nf 46 -g 1 -alpha 0.005 -dp True`
3. Running on MSLRWEB10K dataset: <br> `python main.py -d ./data/MSLR-WEB10K -e 30 -nf 136 -g 1 -alpha 0.005 -dp True`
---

<br>Extra:
---
- MSLRWEB10k Dataset download: https://1drv.ms/u/s!AtsMfWUz5l8nbOIoJ6Ks0bEMp78
- Running these will store the results and the weights with the respective folds in the files `fold_{f}_data` and `fold_{f}_weights` in the current directory.
<br><br>
- Once the main python file has completed running, for visualization use: <br> `python visualizer.py`
  - The fold to visualize is asked as user input. Make sure the fold exists in the same directory as the python file

- Once all folds are run, to know the average and maximum across all folds, run:<br>
`python pickle_reader.py -d ./` 
  - Use -d option to mention the location of all the folds for ex: `python .\pickle_reader.py -d ./OHSUMED_25_analysis/SF2`


