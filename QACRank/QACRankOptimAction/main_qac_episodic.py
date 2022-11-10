from Test import *
from train_qac import *
from LoadData import *
import os
import pickle
import random
import argparse

def pickle_data(data,outfile):
    """Given the data object pickles into path given"""
    if(os.path.exists(outfile)):
        os.remove(outfile)
    
    outhandle=open(outfile,'ab')
    pickle.dump(data, outhandle)
    outhandle.close()

def get_name(datadir):
    """Given data directory returns the dataset name"""
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

    #Parse the arguments
    ap=argparse.ArgumentParser()
    ap.add_argument("-d","--data",required=True,help="Relative or absolute directory of the folder where data dictionary exists.")
    ap.add_argument("-e","--epochs",required=True,help="Number of epochs")
    ap.add_argument("-nf", "--features_no", required=True,help="Number of features in the dataset")
    ap.add_argument("-lr_c", "--learning_rate_critic", required=True,help="critic learning rate")
    ap.add_argument("-lr_a", "--learning_rate_actor", required=True,help="actor learning rate")
    ap.add_argument("-g", "--gamma", required=True,help="gamma value")
    #ap.add_argument("-nstep", "--episodelength", required=True,help="Episode length to consider")
    ap.add_argument("-seed", "--seed", required=True,help="seed for initialization")
    ap.add_argument("-trainsize", "--trainsize", required=True,help="train test split proportion")


    #Initial Settings
    args=vars(ap.parse_args())
    datadir=str(args["data"])
    n_features=int(args["features_no"])
    epochs=int(args["epochs"])
    alpha_actor=float(args['learning_rate_actor'])
    alpha_critic=float(args['learning_rate_critic'])
    gamma=float(args['gamma'])
    #n_step=int(args['episodelength'])
    seed=int(args['seed'])
    trainprop=int(args['trainsize'])
    
    # p='/home/siva/Desktop/Project_Work/MDPRank/Data/OHSUMED/QueryLevelNorm/OHSUMED_All_DIC'
    with open(datadir,'rb') as handle:
        data=pickle.load(handle)
    # n_features=45
    #for fold in [1,2,3,4,5]:
    #Current Fold
    # trainset=datadir+"/Fold"+str(fold)+"/train.txt"
    # validset=datadir+"/Fold"+str(fold)+"/vali.txt"
    # testset=datadir+"/Fold"+str(fold)+"/test.txt"
    # seed=7
    #set the seed
    np.random.seed(seed)
    theta=np.random.randn(n_features,1)
    W=np.random.randn(n_features,1)

    print("\n-------------------Now Running Full Split:---------------------\n")
    theta,W,results=Train_Full_Split_episodicupdate(data,n_features, theta,W,epochs,alpha_actor,alpha_critic,gamma,seed,trainprop)
    #print("**************************Training Complete******************************")
    
    ds=get_name(datadir)
    add_info="allSplit_optimaction_uepisodic_irandn_seed"+str(seed)
    filename=ds+"_e"+str(epochs)+"_lr_a,c,"+str(alpha_actor)+","+str(alpha_critic)+"_g"+str(gamma)+"_"+add_info
    
    #Pickle data for visualization
    outfile="Results/"+ds+"/"+filename+"_data"
    pickle_data(results, outfile)

    #Save weights
    weights={'theta':theta,'W':W}
    outfile="Results/"+ds+"/"+filename+"_weights"    
    pickle_data(weights, outfile)

    #visualize and save plots
    outfile="Results/"+ds+"/"+filename
    #view_save_plot(results, outfile)

    view_save_plot_ndcg(results, outfile)

    print("\n-----------------Complete-------------------------------\n")
