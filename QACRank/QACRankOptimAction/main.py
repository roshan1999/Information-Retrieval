from Test import Test
from train_qac import *
from LoadData import *
import os
import pickle
import random
import argparse

def pickle_data(data,outfile):
    if(os.path.exists(outfile)):
        os.remove(outfile)
    
    outhandle=open(outfile,'ab')
    pickle.dump(data, outhandle)
    outhandle.close()

def get_name(datadir):
    lst=datadir.split('/')
    ds=""
    for i in lst:
        if('ohsumed' in i.lower()):
            ds='OHSUMED'
        elif('mq2008' in i.lower()):
            ds='MQ2008'
        elif('mq2007' in i.lower()):
            ds='MQ2007'
    if(len(ds)==0):
        print("Wrong Dataset,Please check path")
        exit()
    else:
        return ds

if __name__ == '__main__':

    ap=argparse.ArgumentParser()
    ap.add_argument("-d","--data",required=True,help="Relative or absolute directory of the folder where fold folders exist.")
    ap.add_argument("-e","--epochs",required=True,help="Number of epochs for each training process.")
    ap.add_argument("-nf", "--features_no", required=True,help="Number of features in the dataset")
    #ap.add_argument("-lr", "--learning_rate", required=True,help="Learning rate to be used")
    ap.add_argument("-f", "--fold", required=True,help="Fold to run")

    #Initial Settings
    args=vars(ap.parse_args())
    datadir=str(args["data"])
    n_features=int(args["features_no"])
    epochs=int(args["epochs"])
    #lr=float(args["learning_rate"])
    fold=int(args["fold"])
    
    
    #for fold in [1,2,3,4,5]:
    #Current Fold
    trainset=datadir+"/Fold"+str(fold)+"/train.txt"
    validset=datadir+"/Fold"+str(fold)+"/vali.txt"
    testset=datadir+"/Fold"+str(fold)+"/test.txt"
    np.random.seed(fold+1)
    theta = np.random.randn(n_features)*np.sqrt(2/n_features)
    theta = np.round(normalize_minmax(theta) * 2 - 1, 4)
    theta = np.reshape(theta, (n_features, 1))
    # theta = np.round(normalize_minmax(theta) * 2 - 1, 4)
    # p='/home/siva/Desktop/Project_Work/MDPRank/LTR-MDPRank/MQ2008_results(150iters)/fold_4_weights'
    # with open(p,'rb') as handle:
    #     weights=pickle.load(handle)
    # theta=weights
    # theta=np.zeros(n_features)
    # theta = np.reshape(theta, (n_features, 1))
    #Load critic weights
    p='/home/siva/Desktop/Project_Work/MDPRank/LTR-MDPRank/QACRank_MC/Results/MQ2008/MQ2008_f3_e200_a,c,0.05,0.09_g0.95_MCFlavor_weights'
    with open(p,'rb') as handle:
        weights=pickle.load(handle)
    W=weights['W']
    print("Warm Start active")
    # W = np.random.randn(n_features)#*np.sqrt(2/n_features)
    # W = np.reshape(W, (n_features, 1))
    # W = np.round(normalize_minmax(W) * 2 - 1, 4)
    #Hyperparameters
    alpha_actor=0.02
    alpha_critic=0.06
    gamma=0.97
    epochs=epochs
    print("\n-------------------Now running Fold:",fold,"---------------------\n")
    theta,W,results=Train(trainset, validset, testset, n_features, theta,W, epochs,alpha_actor,alpha_critic,gamma)
    print("**************************Training Complete******************************")
    
    #Testing
    print("\nTesting Results:")
    ndcg_values=Test(theta, testset, n_features)
    print(f"NDCG mean value = {ndcg_values[0]} NDCG@1 mean value = {ndcg_values[1]} NDCG@3 mean value = {ndcg_values[2]} NDCG@5 mean value = {ndcg_values[3]} NDCG@10 mean value = {ndcg_values[4]}")

    ds=get_name(datadir)
    add_info="MCFlavor-warmstart200,ac-irandnmm"
    filename=ds+"_f"+str(fold)+"_e"+str(epochs)+"_a,c,"+str(alpha_actor)+","+str(alpha_critic)+"_g"+str(gamma)+"_"+add_info
    
    #Pickle data for visualization
    outfile="Results/"+ds+"/full_try/try3/"+filename+"_data"
    pickle_data(results, outfile)

    #Save weights
    weights={'theta':theta,'W':W}
    outfile="Results/"+ds+"/full_try/try3/"+filename+"_weights"    
    pickle_data(weights, outfile)

    print("\n-----------------Fold:",fold,"Complete-------------------------------\n")
