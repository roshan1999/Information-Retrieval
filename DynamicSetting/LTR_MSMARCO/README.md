# Dynamic Ranking Simulator
### Files to download
---
1. [Corpus document set](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz)
2. [Corpus Document offset lookup](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs-lookup.tsv.gz)
3. [Query with ID](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz)
4. [Doc2Vec trained model (download all files)](https://drive.google.com/drive/folders/1nB_XAhbtvBSoi8jzXhZvfVQLhTRoiEqo?usp=sharing)


<br>PreRequisites:
```
   gensim: pip install gensim
   pytorch
   nltk: pip install nltk
   scipy: pip install scipy
   numpy: pip install numpy
   pandas: pip install pandas
   matplotlib: pip install matplotlib
```


> Note: Extract 'msmarco-docs.tsv.gz' to get 'msmarco-docs.tsv'

These files must finally exist in the directory:

Directory Structure post download and extraction:

![Directory Structure](https://user-images.githubusercontent.com/34297593/125894798-1c638d4e-28b5-44d8-8e66-c9bcfc5ab82d.png)



To run the algorithm, enter the following code:<br>
`$ python visual.py -M init_docs -extra add_docs -d decay_value -nf no_features -g gamma -alpha hyperparam1 -beta hyperparam2 -lr learning_rate -window window_size -size corpus_size`<br>

Example : <br>

`$ python ./visual.py -M 10000 -d 0.01 -extra 1000`<br>

> Note: This runs the algorithm for initialized 10000 documents with extra 1000 documents at each time step added. The decay is 0.01

<br>Default hyperparameters: 
---
1. Initial M documents (-M) : `Required`
2. Extra documents to add at each time step (-extra) : `Required`
3. Decay(-d): `Required`
4. Number of features(-nf): `50`
6. Gamma(-g): `1`
7. Learning rate(-lr): `0.001`
8. Hyper Parameter SS retrieval(-alpha): `0.1`
8. Hyper Parameter RS retrieval(-beta): `0.01`
9. Sliding Window Size(-window): `50000`
10. Entire corpus size(-size): `200000`



