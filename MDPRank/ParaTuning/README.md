## MDPRank
---


Implementation of MDPRank<br>
To run the algorithm, enter the following code:<br>
`$ python main.py -d datapath -nf number_features -e no_epochs`<br>

<br>For example: 
---
1. Running on MQ2008 dataset: <br> `$ python main.py -d /home/siva/Desktop/Project_Work/MDPRank/Data/MQ2008 -e 2 -nf 46`
2. Running on OHSUMED 45 dataset: <br> `$ python main.py -d /home/siva/Desktop/Project_Work/MDPRank/Data/OHSUMED/QueryLevelNorm -e 2 -nf 45`
3. Running on MQ2007 dataset: <br> `$ python main.py -d /home/siva/Desktop/Project_Work/MDPRank/Data/MQ2007 -e 2 -nf 46`


<br>
Each Fold has a trainset,validationset and testset.<br>
In each fold, multiple models with different hyperparameters are trained on the trainset,<br>
validationset is used to select the best model and test set is used to evaluate the best model.<br>
<br>
This process is carried out for each of the 5 folds.<br>
Results are saved in fold_foldno_data and weights in fold_foldno_weights for further analysis.<br>
<br>

---

<br>Further Analysis:


- Can visualize by running: <br> `$ python visualizer.py -f foldnumber`<br>
For example: 
Visualizing 1st fold <br> `$ python visualizer.py -f 1` 

- To see the average and max over all the folds run:
`python averager.py`<br>

---
