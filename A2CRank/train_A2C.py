import numpy as np
from Eval import evaluate,evaluate_labels
from LoadData import import_dataset
from Utils import *
from Test import *
import time

def Train_Full_Split_MCCritic(data,n_features, theta,W,epochs,alpha_actor,alpha_critic,gamma,seed,nstep,trainprop):
    """Trains policy for given number of epochs using theta,W as starting point"""

    gamma = gamma
    alpha_theta=alpha_actor
    alpha_w= alpha_critic
    trainprop=trainprop
    len_episode=nstep

    #Import dataset and perform the train-test-split
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

    #Results dictionary: Has train mean_ndcgs,total-reward,avg-step-reward,test mean_ndcgs across the current fold
    results={'train':[],'test':[],'train_episode_rewards':[]}

    print("Initial Training Scores:")
    ndcg_values = Test_Queries(theta, data,train_queries_list,n_features)
    print(f"NDCG mean value = {ndcg_values[0]} NDCG@1 mean value = {ndcg_values[1]} NDCG@3 mean value = {ndcg_values[2]} NDCG@5 mean value = {ndcg_values[3]} NDCG@10 mean value = {ndcg_values[4]}")
    results['train'].append(tuple(ndcg_values))

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
            #Buffers
            buf_rewards=[]
            buf_phi_s=[]
            buf_v_s=[]
            buf_score_s=[]
            epi_total_r=0
            epi_step_r=0
            
            
            #Initialize state
            t=0
            docidsleft=list(data[q].keys())
            M=len(docidsleft)
            s=[t,docidsleft.copy()]
            effective_length=min(M-1,len_episode)

            #train labels
            labels=[]

            while(t!=min(M,len_episode)):
                #print("t:",t,"s:",len(s[1]),"probn:",len(probs))

                #Initial Action 
                a,probs=sample_action(theta,s,data,q)
                #a=sample_best_action(s,data,q)
                labels.append(data[q][a][1])
                
                #Get reward
                r=get_reward(a,t,data,q)
                buf_rewards.append(r)
                r_epoch_total+=r
                epi_total_r+=r
                epi_step_r+=r
                
                #Find the scorefn of s
                scorefn_s_a=compute_scorefn(s,a,theta,data,q)
                buf_score_s.append(scorefn_s_a)
                
                #Construct phi(s) vector
                phi_s=get_phi(probs,s,data,q)
                buf_phi_s.append(phi_s)
            
                #Find the V values
                v_s_a=compute_vvalue(W,phi_s)#Compute the v(s)
                buf_v_s.append(v_s_a)
                
                #Next state
                t=t+1
                if(t==M):
                    break
                docidsleft.pop(docidsleft.index(a))
                s_next=[t,docidsleft.copy()]
                
                #Change state variables
                s=s_next

            #Step Performance
            epi_step_r=epi_step_r/effective_length #How good 1 step is in current episode
            results['train_episode_rewards'].append(epi_step_r)
            r_epoch_step_mean+=epi_step_r

            #Update parameters

            #Compute all t step returns
            returns = []

            #Bootstrap if episode truncated
            if(t==M):
                dis_reward = 0
            else:
                dis_reward = buf_v_s[-1]
            
            #Compute returns
            for reward in buf_rewards[::-1]:
                dis_reward = reward + gamma * dis_reward
                returns.insert(0, dis_reward)
            
            #Update the Weights
            for t in range(min(M,len_episode)):
                theta=theta+alpha_actor*buf_score_s[t]*(returns[t]-buf_v_s[t])
                W=W+ alpha_critic* (returns[t]-buf_v_s[t]) * buf_phi_s[t]
            
            #Normalize the weights
            if(np.linalg.norm(theta)!=0):
                theta=theta/np.linalg.norm(theta)
            if(np.linalg.norm(W)!=0):
                W=W/np.linalg.norm(W)
            
            #Training NDCGS for current query
            NDCG_values.append(evaluate_labels(labels))
            NDCG_1_values.append(evaluate_labels(labels,1))
            NDCG_3_values.append(evaluate_labels(labels,3))
            NDCG_5_values.append(evaluate_labels(labels,5))
            NDCG_10_values.append(evaluate_labels(labels,10))

        #Evaluate the training
        print("Total Reward(across all querys):",np.round(r_epoch_total,5))
        print("Avg Reward(per step):",np.round(r_epoch_step_mean/n_queries,5))
            
        # After completion of all queries:Mean NDCGs
        NDCG_mean = np.mean(np.array(NDCG_values))
        NDCG_1_mean = np.mean(np.array(NDCG_1_values))
        NDCG_3_mean = np.mean(np.array(NDCG_3_values))
        NDCG_5_mean = np.mean(np.array(NDCG_5_values))
        NDCG_10_mean = np.mean(np.array(NDCG_10_values))

        #Updating results dictionary
        results['train'].append((NDCG_mean, NDCG_1_mean, NDCG_3_mean,NDCG_5_mean, NDCG_10_mean
            ,np.round(r_epoch_total,5),np.round(r_epoch_step_mean/n_queries,5)))

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
