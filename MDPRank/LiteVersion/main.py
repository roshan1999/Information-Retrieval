from Test import Test
from Train import *
from LoadData import normalize_std, normalize_minmax
import os
import pickle
import random
import argparse

def pickle_data(data,outfile):
    """"Pickle the python objects into the output_file"""""
    if(os.path.exists(outfile)):
        os.remove(outfile)
    
    outhandle=open(outfile,'ab')
    pickle.dump(data, outhandle)
    outhandle.close()


if __name__ == '__main__':

    ap=argparse.ArgumentParser()
    ap.add_argument("-d","--data",required=True,help="Relative or absolute directory of the folder where fold folders exist.")
    ap.add_argument("-e","--epochs",required=True,help="Number of epochs for each training process.")
    ap.add_argument("-nf", "--features_no", required=True,help="Number of features in the dataset")
    ap.add_argument("-lr", "--learning_rate", required=True,help="Learning rate to be used")
    ap.add_argument("-f", "--fold", required=True,help="Fold to run")

    #Initial Settings
    args=vars(ap.parse_args())
    datadir=str(args["data"])
    n_features=int(args["features_no"])
    epochs=int(args["epochs"])
    lr=float(args["learning_rate"])
    fold=int(args["fold"])
    
    
    #for fold in [1,2,3,4,5]:
    #Current Fold
    trainset=datadir+"/Fold"+str(fold)+"/train.txt"
    validset=datadir+"/Fold"+str(fold)+"/vali.txt"
    testset=datadir+"/Fold"+str(fold)+"/test.txt"
    np.random.seed(fold)
    W = np.random.randn(n_features)*np.sqrt(2/n_features)
    W = np.reshape(W, (n_features, 1))
    W = np.round(normalize_minmax(W) * 2 - 1, 4)

    print("\n-------------------Now running Fold:",fold,"---------------------\n")
    w,result=Train(trainset, validset, testset, n_features, W, epochs,lr)
    best_W=w
    best_results=result
    #Find best lr using validation set
    #lr=[0.01,0.05,0.09,0.001]
    # results_alpha=[]
    # W_alpha=[]
    # results=[]
    # for alpha in lr:
    #     print("\nFor lr =",alpha,"\n")
    #     w,result=Train(trainset, validset, testset, n_features, W, epochs,alpha)
    #     W_alpha.append(w)
    #     results.append(result)
    #     print("Validation:")
    #     ndcg_values = Test(W, validset,n_features)
    #     print(
    #         f"NDCG mean value = {ndcg_values[0]} NDCG@1 mean value = {ndcg_values[1]} NDCG@3 mean value = {ndcg_values[2]} NDCG@5 mean value = {ndcg_values[3]} NDCG@10 mean value = {ndcg_values[4]}")
    #     results_alpha.append(np.mean(ndcg_values[1:]))
    print("**************************Training Complete******************************")
    
    #Pick Best Model
    # best_lr=lr[np.argmax(results_alpha)]
    # print("\nBest alpha:",best_lr)
    # best_W=W_alpha[np.argmax(results_alpha)]
    # best_results=results[np.argmax(results_alpha)]
    
    #Testing
    print("\nTesting Results:")
    ndcg_values=Test(best_W, testset, n_features)
    print(f"NDCG mean value = {ndcg_values[0]} NDCG@1 mean value = {ndcg_values[1]} NDCG@3 mean value = {ndcg_values[2]} NDCG@5 mean value = {ndcg_values[3]} NDCG@10 mean value = {ndcg_values[4]}")

    #Pickle data for visualization
    outfile="Results/fold_"+str(fold)+"_data"
    pickle_data(best_results, outfile)

    #Save weights
    outfile="Results/fold_"+str(fold)+"_weights"
    pickle_data(best_W, outfile)

    print("\n-----------------Fold:",fold,"Complete-------------------------------\n")
