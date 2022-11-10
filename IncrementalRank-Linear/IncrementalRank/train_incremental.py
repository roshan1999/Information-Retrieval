import numpy as np
from Eval import evaluate,evaluate_labels
from LoadData import import_dataset
from Utils import *
from Test import *
import time

def Train_Full_Split_Incremental(data,n_features,theta,W,epochs,alpha_actor,alpha_critic,gamma,seed,n_step,trainprop):
    """Trains dataset for given number of iterations using W as starting point and validates after each epoch"""

    gamma = gamma
    alpha_theta=alpha_actor
    alpha_w= alpha_critic
    len_episode=n_step

    #Import dataset and split into train and test
    data = data
    np.random.seed(seed)
    trainsize=int(len(data)*(trainprop/100))
    testsize=len(data)-trainsize
    train_queries_list=list(np.random.choice(list(data.keys()),size=trainsize,replace=False))
    test_queries_list=[]
    for qid in data:
        if(qid not in train_queries_list):
            test_queries_list.append(qid)     
    n_queries=trainsize
    #Results dictionary: Has train mean_ndcgs,total-reward,avg-reward,validation mean_ndcgs,test mean_ndcgs across the current fold
    results={'training':[],'test':[]}

    print("Initial Training Scores:")
    ndcg_values = Test_Queries(theta, data,train_queries_list,n_features)
    print(f"NDCG mean value = {ndcg_values[0]} NDCG@1 mean value = {ndcg_values[1]} NDCG@3 mean value = {ndcg_values[2]} NDCG@5 mean value = {ndcg_values[3]} NDCG@10 mean value = {ndcg_values[4]}")
    results['training'].append(tuple(ndcg_values))

    # # Printing Testing Scores
    print("\nInitial Test Scores:")
    ndcg_values = Test_Queries(theta, data,test_queries_list,n_features)
    print(f"NDCG mean value = {ndcg_values[0]} NDCG@1 mean value = {ndcg_values[1]} NDCG@3 mean value = {ndcg_values[2]} NDCG@5 mean value = {ndcg_values[3]} NDCG@10 mean value = {ndcg_values[4]}")
    results['test'].append(tuple(ndcg_values))
    print()

    #Each timestep update
    for i in range(1,epochs+1):
        e_start=time.time()

        # print("\n***********EPOCH: " + str(i) + " START***********\n")
        NDCG_values = []
        NDCG_1_values = []
        NDCG_3_values = []
        NDCG_5_values = []
        NDCG_10_values = []

        #Monitor
        r_epoch_total=0
        r_epoch_step_mean=0

        for q in train_queries_list:
            #Initialize state
            t=0
            docidsleft=list(data[q].keys())
            M=len(docidsleft)
            s=[t,docidsleft.copy()]
            epi_step_r=0
            #Monitors
            labels=[]
            
            #Take Action 
            a,probs=sample_action(theta,s,data,q)
            labels.append(data[q][a][1])

            effective_length=min(M,len_episode)
            
            while(t!=min(M-1,len_episode)):
                #print("t:",t,"s:",len(s[1]),"probn:",len(probs))
                #Get reward
                r=get_reward(a,t,data,q)
                r_epoch_total+=r
                epi_step_r+=r
                
                #Change state
                t=t+1
                docidsleft.pop(docidsleft.index(a))
                s_next=[t,docidsleft.copy()]
                
                #Take Action
                a_next,probs_next=sample_action(theta,s_next,data,q)
                labels.append(data[q][a_next][1])
                
                #Find the scorefn of s
                scorefn_s_a=compute_scorefn(s,a,theta,data,q)
                
                #Construct phi(s) vector
                phi_s=get_phi(probs,s,data,q)
                phi_s_next=get_phi(probs_next,s_next,data,q)
            
                #Find the V values
                v_s_a=compute_vvalue(W,phi_s)#Compute the q(s,a)
                v_s_next=compute_vvalue(W,phi_s_next)#Compute q(s_next,a_next)


                #td(0) error
                td_error=r+gamma*v_s_next-v_s_a
                
                #updates
                theta+=alpha_theta*scorefn_s_a*td_error #Policy Update
                W+= alpha_w*td_error*phi_s #Critic Update
                
                #Normalize weights
                theta = theta/np.linalg.norm(theta)
                W=W/np.linalg.norm(W)
                
                #Change state variables
                s=s_next
                a=a_next
                probs=probs_next

            r_epoch_step_mean+=epi_step_r/effective_length
            #print("Queryid:",q," thetadelta:",theta_acc," Wacc:",W_acc)
            #Training NDCGS for current query
            NDCG_values.append(evaluate_labels(labels))
            NDCG_1_values.append(evaluate_labels(labels,1))
            NDCG_3_values.append(evaluate_labels(labels,3))
            NDCG_5_values.append(evaluate_labels(labels,5))
            NDCG_10_values.append(evaluate_labels(labels,10))
        

        #Evaluate the training
        print("Total Reward(across all querys):",np.round(r_epoch_total,5))
        print("Avg Reward(per query):",np.round(r_epoch_total/n_queries,5))
        print("Avg Reward(per step):",np.round(r_epoch_step_mean/n_queries,5))
            
        # After completion of all queries:Mean NDCGs
        NDCG_mean = np.mean(np.array(NDCG_values))
        NDCG_1_mean = np.mean(np.array(NDCG_1_values))
        NDCG_3_mean = np.mean(np.array(NDCG_3_values))
        NDCG_5_mean = np.mean(np.array(NDCG_5_values))
        NDCG_10_mean = np.mean(np.array(NDCG_10_values))

        #Updating results dictionary
        results['training'].append((NDCG_mean, NDCG_1_mean, NDCG_3_mean,NDCG_5_mean, NDCG_10_mean,np.round(r_epoch_step_mean/n_queries,7),
            np.round(r_epoch_total,5),np.round(r_epoch_total/n_queries,5)))

        #Printing Training Scores on the terminal
        print("Training Scores:")
        print(
            f"NDCG mean value = {NDCG_mean} NDCG@1 mean value = {NDCG_1_mean} NDCG@3 mean value = {NDCG_3_mean} NDCG@5 mean value = {NDCG_5_mean} NDCG@10 mean value = {NDCG_10_mean}")

        # # Printing Testing Scores
        print("\nTest Scores:")
        ndcg_values = Test_Queries(theta, data,test_queries_list,n_features)
        print(f"NDCG mean value = {ndcg_values[0]} NDCG@1 mean value = {ndcg_values[1]} NDCG@3 mean value = {ndcg_values[2]} NDCG@5 mean value = {ndcg_values[3]} NDCG@10 mean value = {ndcg_values[4]}")
        results['test'].append(tuple(ndcg_values))
        
        print("time-taken:",np.round(time.time()-e_start,4),"s")
        print("\n***********EPOCH: " + str(i) + " COMPLETE***********\n")
        
    return theta,W,results