from Test import *
from train_A2C import *
from LoadData import *
import os
import pickle
import random
import argparse

def pickle_data(data,outfile):
    """Pickles given data object into an outfile"""
    if(os.path.exists(outfile)):
        os.remove(outfile)
    
    outhandle=open(outfile,'ab')
    pickle.dump(data, outhandle)
    outhandle.close()

def get_name(datadir):
    """Gets name of dataset from the path"""
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
    ap.add_argument("-d","--data",required=True,help="Relative or absolute directory of the folder where data dictionary exists.")
    ap.add_argument("-e","--epochs",required=True,help="Number of epochs")
    ap.add_argument("-nf", "--features_no", required=True,help="Number of features in the dataset")
    ap.add_argument("-lr_critic", "--learning_rate_critic", required=True,help="critic learning rate")
    ap.add_argument("-lr_actor", "--learning_rate_actor", required=True,help="actor learning rate")
    ap.add_argument("-g", "--gamma", required=True,help="gamma value")
    ap.add_argument("-episode_length", "--episodelength", required=True,help="Episode length to consider")
    ap.add_argument("-seed", "--seed", required=True,help="seed for initialization")
    ap.add_argument("-trainsize", "--trainsize", required=False,default=70,help="train test split proportion")
    ap.add_argument("-smooth", "--smooth", required=False,default=800,help="number of episodes to smooth graph over")


    #Initial Settings
    args=vars(ap.parse_args())
    datadir=str(args["data"])
    n_features=int(args["features_no"])
    epochs=int(args["epochs"])
    alpha_actor=float(args['learning_rate_actor'])
    alpha_critic=float(args['learning_rate_critic'])
    gamma=float(args['gamma'])
    n_step=int(args['episodelength'])
    seed=int(args['seed'])
    trainprop=int(args['trainsize'])
    smooth=int(args['smooth'])
    
    # Hyperparameters
    # alpha_actor=0.0003
    # alpha_critic=0.0004
    # gamma=1
    # epochs=epochs
    # n_step=20
    # datadir='/home/siva/Desktop/Project_Work/MDPRank/Data/OHSUMED/QueryLevelNorm/OHSUMED_All_DIC'
    # n_features=45
    # seed=7
    # trainprop=70

    #Load the data dictionary
    with open(datadir,'rb') as handle:
        data=pickle.load(handle)

    #Set the random seed
    np.random.seed(seed)

    #Initialize weights
    
    theta=np.random.randn(n_features,1)
    W=np.random.randn(n_features,1)

    # theta = np.random.rand(n_features)
    # theta= 2*normalize_minmax(theta)-1
    # theta = np.reshape(theta, (n_features, 1))
    
    # theta=np.zeros((n_features,1))
    # W=np.zeros((n_features,1))

    print("\n-------------------Now Running Full Split:---------------------\n")
    print("Hyperparameters:","\nalpha_actor:",alpha_actor," alpha_critic:",alpha_critic," gamma:",gamma," epochs:",epochs," seed:",seed,"\n\n")
    theta,W,results=Train_Full_Split_MCCritic(data,n_features,theta,W,epochs,alpha_actor,alpha_critic,gamma,seed,n_step,trainprop)
    #print("**************************Training Complete******************************")
    
    #frame the file name including all details of the run
    ds=get_name(datadir)
    add_info="allSplit-a2c-iac_randn-length_"+str(n_step)+"-seed"+str(seed)+"-trainprop_"+str(trainprop)
    filename=ds+"-e"+str(epochs)+"-a,c,"+str(alpha_actor)+","+str(alpha_critic)+"-g"+str(gamma)+"-"+add_info
    
    #Pickle data for visualization
    outfile="Results/"+ds+"/"+filename+"_data"
    pickle_data(results, outfile)

    #Save weights
    weights={'theta':theta,'W':W}
    outfile="Results/"+ds+"/"+filename+"_weights"    
    pickle_data(weights, outfile)

    #visualize/save figures
    
    #Avg step return vs epoch
    outfile="Results/"+ds+"/"+filename
    view_save_plot_avgreturn(results, outfile)
    
    #NDCG Graph
    view_save_plot_ndcg(results, outfile)
    
    #Avg step return vs episode
    # smooth=smooth
    # outfile="Results/"+ds+"/"+filename+"-smooth_"+str(smooth)
    # view_save_plot_episode(results,smooth,outfile)

    print("\n-----------------Complete-------------------------------\n")
