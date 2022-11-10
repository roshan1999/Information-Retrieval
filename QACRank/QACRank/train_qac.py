import numpy as np
from Eval import evaluate,evaluate_labels
from LoadData import import_dataset
from QACUtils import *
from Test import *
import time


def Train(trainset,validset,testset,n_features, theta,W,epochs,alpha_actor,alpha_critic,gamma):
    """Trains dataset for given number of iterations using W as starting point and validates after each epoch"""

    gamma = gamma
    alpha_theta=alpha_actor
    alpha_w= alpha_critic
    n_queries=100
    len_episode=20

    #Import dataset
    data = import_dataset(trainset,n_features)

    #Results dictionary: Has train mean_ndcgs,total-reward,avg-reward,validation mean_ndcgs,test mean_ndcgs across the current fold
    results={'training':[],'validation':[],'test':[]}

    #Each Epoch does a weight update based on trainset and then test
    for i in range(1,epochs+1):
        e_start=time.time()

        queries_list=np.random.choice(list(data.keys()),size=n_queries)

        # print("\n***********EPOCH: " + str(i) + " START***********\n")
        NDCG_values = []
        NDCG_1_values = []
        NDCG_3_values = []
        NDCG_5_values = []
        NDCG_10_values = []

        #Monitor
        G_total=0
        theta_acc=np.zeros((n_features,1))
        W_acc=np.zeros((n_features,1))
        for q in queries_list:
            #Initialize state
            t=0
            docidsleft=list(data[q].keys())
            M=len(docidsleft)
            s=[t,docidsleft.copy()]
            #Monitors
            labels=[]
            #Take Action 
            a=sample_action(theta,s,data,q)
            labels.append(data[q][a][1])
            while(t!=M-2):
                #Get reward
                r=get_reward(a,t,data,q)
                G_total+=r
                #Change state
                t=t+1
                docidsleft.pop(docidsleft.index(a))
                s_next=[t,docidsleft.copy()]
                #Take Action
                a_next=sample_action(theta,s_next,data,q)
                labels.append(data[q][a_next][1])
                #Find the scorefn of s,s_new
                grad_s_a=compute_scorefn(s,a,theta,data,q)
                grad_s_next_a_next=compute_scorefn(s_next,a_next,theta,data,q)
                #Add dimension to form psi feature vector
                psi_s_a=grad_s_a#add_dim(grad_s_a,t-1)
                psi_s_next_a_next=grad_s_next_a_next#add_dim(grad_s_next_a_next,t)
                #Find the Q values
                q_s_a=compute_qvalue(W,psi_s_a)#Compute the q(s,a)
                q_s_next_a_next=compute_qvalue(W,psi_s_next_a_next)#Compute q(s_next,a_next)
                #updates
                theta_acc+=alpha_theta*grad_s_a*q_s_a #Policy Update
                td_error=r+gamma*q_s_next_a_next - q_s_a #td(0) error
                W_acc+= alpha_w*td_error*psi_s_a #Critic Update
                #Normalize weights
                #theta = theta/np.linalg.norm(theta)
                #W=W/np.linalg.norm(W)
                #Change state variables
                s=s_next
                a=a_next
            #print("Queryid:",q," thetadelta:",theta_acc," Wacc:",W_acc)
            #Training NDCGS for current query
            NDCG_values.append(evaluate_labels(labels))
            NDCG_1_values.append(evaluate_labels(labels,1))
            NDCG_3_values.append(evaluate_labels(labels,3))
            NDCG_5_values.append(evaluate_labels(labels,5))
            NDCG_10_values.append(evaluate_labels(labels,10))
        
        #Updates
        theta+=theta_acc/n_queries
        W+=W_acc/n_queries
        #Normalize
        theta = theta/np.linalg.norm(theta)
        W = W/np.linalg.norm(W)

        #Evaluate the training
        print("Total Return(across all querys):",np.round(G_total,5))
        print("Avg Return(per query):",np.round(G_total/n_queries,5))
            
        # After completion of all queries:Mean NDCGs
        NDCG_mean = np.mean(np.array(NDCG_values))
        NDCG_1_mean = np.mean(np.array(NDCG_1_values))
        NDCG_3_mean = np.mean(np.array(NDCG_3_values))
        NDCG_5_mean = np.mean(np.array(NDCG_5_values))
        NDCG_10_mean = np.mean(np.array(NDCG_10_values))

        #Updating results dictionary
        results['training'].append((NDCG_mean, NDCG_1_mean, NDCG_3_mean,NDCG_5_mean, NDCG_10_mean,np.round(G_total,5),np.round(G_total/n_queries,5)))

        #Printing Training Scores on the terminal
        print("Training Scores:")
        print(
            f"NDCG mean value = {NDCG_mean} NDCG@1 mean value = {NDCG_1_mean} NDCG@3 mean value = {NDCG_3_mean} NDCG@5 mean value = {NDCG_5_mean} NDCG@10 mean value = {NDCG_10_mean}")

        
        # # Printing Validation Scores
        print("\nValidation Scores:")
        ndcg_values = Test(theta, validset,n_features)
        print(f"NDCG mean value = {ndcg_values[0]} NDCG@1 mean value = {ndcg_values[1]} NDCG@3 mean value = {ndcg_values[2]} NDCG@5 mean value = {ndcg_values[3]} NDCG@10 mean value = {ndcg_values[4]}")
        results['validation'].append(tuple(ndcg_values))

        # # Printing Testing Scores
        print("\nTest Scores:")
        ndcg_values = Test(theta, testset,n_features)
        print(f"NDCG mean value = {ndcg_values[0]} NDCG@1 mean value = {ndcg_values[1]} NDCG@3 mean value = {ndcg_values[2]} NDCG@5 mean value = {ndcg_values[3]} NDCG@10 mean value = {ndcg_values[4]}")
        results['test'].append(tuple(ndcg_values))
        
        print("time-taken:",np.round(time.time()-e_start,4),"s")
        print("\n***********EPOCH: " + str(i) + " COMPLETE***********\n")
        
    return theta,W,results

def Train_Full_Split_stepupdate(data,n_features, theta,W,epochs,alpha_actor,alpha_critic,gamma,seed,trainprop):
    """Trains policy for given number of iterations using theta,W as starting point """

    gamma = gamma
    alpha_theta=alpha_actor
    alpha_w= alpha_critic
    # n_queries=100
    # len_episode=20

    #Import dataset and split into train and test sets
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

    #Results dictionary initialization
    results={'training':[],'test':[]}

    #Initial Values
    print("Initial Training Scores:")
    ndcg_values = Test_Queries(theta, data,train_queries_list,n_features)
    print(f"NDCG mean value = {ndcg_values[0]} NDCG@1 mean value = {ndcg_values[1]} NDCG@3 mean value = {ndcg_values[2]} NDCG@5 mean value = {ndcg_values[3]} NDCG@10 mean value = {ndcg_values[4]}")
    results['training'].append(tuple(ndcg_values))

    print("\nInitial Test Scores:")
    ndcg_values = Test_Queries(theta, data,test_queries_list,n_features)
    print(f"NDCG mean value = {ndcg_values[0]} NDCG@1 mean value = {ndcg_values[1]} NDCG@3 mean value = {ndcg_values[2]} NDCG@5 mean value = {ndcg_values[3]} NDCG@10 mean value = {ndcg_values[4]}")
    results['test'].append(tuple(ndcg_values))
    print()
    
    
    #Main training loop
    for i in range(1,epochs+1):
        #Monitor time taken
        e_start=time.time()

        # print("\n***********EPOCH: " + str(i) + " START***********\n")
        NDCG_values = []
        NDCG_1_values = []
        NDCG_3_values = []
        NDCG_5_values = []
        NDCG_10_values = []

        #Monitors
        G_total=0
        r_epoch_step_mean=0

        for q in train_queries_list:

            #Initialize state
            t=0
            docidsleft=list(data[q].keys())
            M=len(docidsleft)
            s=[t,docidsleft.copy()]
            
            #Monitors
            epi_step_r=0
            labels=[]

            #Take Action 
            a=sample_action(theta,s,data,q)
            labels.append(data[q][a][1])
            
            effective_length=M

            while(t!=M-1):
                #Get reward
                r=get_reward(a,t,data,q)
                G_total+=r
                epi_step_r+=r

                #Change state
                t=t+1
                docidsleft.pop(docidsleft.index(a))
                s_next=[t,docidsleft.copy()]
                
                #Take Action
                a_next=sample_action(theta,s_next,data,q)
                labels.append(data[q][a_next][1])
                
                #Find the scorefn of s,s_new
                grad_s_a=compute_scorefn(s,a,theta,data,q)
                grad_s_next_a_next=compute_scorefn(s_next,a_next,theta,data,q)
                
                #define the psi feature vector
                psi_s_a=grad_s_a
                psi_s_next_a_next=grad_s_next_a_next
                
                #Find the Q values
                q_s_a=compute_qvalue(W,psi_s_a)#Compute the q(s,a)
                q_s_next_a_next=compute_qvalue(W,psi_s_next_a_next)#Compute q(s_next,a_next)
                
                #updates
                theta+=alpha_theta*grad_s_a*q_s_a #Policy Update
                td_error=r+gamma*q_s_next_a_next - q_s_a #td(0) error
                W+= alpha_w*td_error*psi_s_a #Critic Update
                
                #Normalize weights
                theta = theta/np.linalg.norm(theta)
                W=W/np.linalg.norm(W)
                
                #Change state variables
                s=s_next
                a=a_next

            r_epoch_step_mean+=epi_step_r/effective_length
            
            #Training NDCGS for current query
            #print("\nlabels:",labels)
            NDCG_values.append(evaluate_labels(labels))
            NDCG_1_values.append(evaluate_labels(labels,1))
            NDCG_3_values.append(evaluate_labels(labels,3))
            NDCG_5_values.append(evaluate_labels(labels,5))
            NDCG_10_values.append(evaluate_labels(labels,10))


        #Evaluate the training
        print("Total Return(across all querys):",np.round(G_total,5))
        print("Avg Return(per query):",np.round(G_total/n_queries,5))
        print("Avg Reward(per step):",np.round(r_epoch_step_mean/n_queries,5))
            
        # After completion of all queries:Mean NDCGs
        NDCG_mean = np.mean(np.array(NDCG_values))
        NDCG_1_mean = np.mean(np.array(NDCG_1_values))
        NDCG_3_mean = np.mean(np.array(NDCG_3_values))
        NDCG_5_mean = np.mean(np.array(NDCG_5_values))
        NDCG_10_mean = np.mean(np.array(NDCG_10_values))

        #Updating results dictionary
        results['training'].append((NDCG_mean, NDCG_1_mean, NDCG_3_mean,NDCG_5_mean, NDCG_10_mean,np.round(r_epoch_step_mean/n_queries,7),np.round(G_total,5),np.round(G_total/n_queries,5)))

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

def Train_Full_Split_epochupdate(data,n_features, theta,W,epochs,alpha_actor,alpha_critic,gamma,seed,trainprop):
    """Trains policy for given number of iterations using theta,W as starting point """

    gamma = gamma
    alpha_theta=alpha_actor
    alpha_w= alpha_critic
    # n_queries=100
    # len_episode=20

    #Import dataset and split into train and test sets
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

    #Results dictionary initialization
    results={'training':[],'test':[]}

    #Initial Values
    print("Initial Training Scores:")
    ndcg_values = Test_Queries(theta, data,train_queries_list,n_features)
    print(f"NDCG mean value = {ndcg_values[0]} NDCG@1 mean value = {ndcg_values[1]} NDCG@3 mean value = {ndcg_values[2]} NDCG@5 mean value = {ndcg_values[3]} NDCG@10 mean value = {ndcg_values[4]}")
    results['training'].append(tuple(ndcg_values))

    print("\nInitial Test Scores:")
    ndcg_values = Test_Queries(theta, data,test_queries_list,n_features)
    print(f"NDCG mean value = {ndcg_values[0]} NDCG@1 mean value = {ndcg_values[1]} NDCG@3 mean value = {ndcg_values[2]} NDCG@5 mean value = {ndcg_values[3]} NDCG@10 mean value = {ndcg_values[4]}")
    results['test'].append(tuple(ndcg_values))
    print()
    
    
    #Main training loop
    for i in range(1,epochs+1):
        #Monitor time taken
        e_start=time.time()

        # print("\n***********EPOCH: " + str(i) + " START***********\n")
        NDCG_values = []
        NDCG_1_values = []
        NDCG_3_values = []
        NDCG_5_values = []
        NDCG_10_values = []

        #Monitors
        G_total=0
        r_epoch_step_mean=0

        #initialize buffers
        theta_acc=np.zeros((n_features,1))
        W_acc=np.zeros((n_features,1))

        for q in train_queries_list:

            #Initialize state
            t=0
            docidsleft=list(data[q].keys())
            M=len(docidsleft)
            s=[t,docidsleft.copy()]
            
            #Monitors
            epi_step_r=0
            labels=[]

            #Take Action 
            a=sample_action(theta,s,data,q)
            labels.append(data[q][a][1])
            
            effective_length=M

            while(t!=M-1):
                #Get reward
                r=get_reward(a,t,data,q)
                G_total+=r
                epi_step_r+=r

                #Change state
                t=t+1
                docidsleft.pop(docidsleft.index(a))
                s_next=[t,docidsleft.copy()]
                
                #Take Action
                a_next=sample_action(theta,s_next,data,q)
                labels.append(data[q][a_next][1])
                
                #Find the scorefn of s,s_new
                grad_s_a=compute_scorefn(s,a,theta,data,q)
                grad_s_next_a_next=compute_scorefn(s_next,a_next,theta,data,q)
                
                #define the psi feature vector
                psi_s_a=grad_s_a
                psi_s_next_a_next=grad_s_next_a_next
                
                #Find the Q values
                q_s_a=compute_qvalue(W,psi_s_a)#Compute the q(s,a)
                q_s_next_a_next=compute_qvalue(W,psi_s_next_a_next)#Compute q(s_next,a_next)
                
                #updates
                theta_acc+=alpha_theta*grad_s_a*q_s_a #Policy Update
                td_error=r+gamma*q_s_next_a_next - q_s_a #td(0) error
                W_acc+= alpha_w*td_error*psi_s_a #Critic Update
                
                #Change state variables
                s=s_next
                a=a_next

            r_epoch_step_mean+=epi_step_r/effective_length
            
            #Training NDCGS for current query
            #print("\nlabels:",labels)
            NDCG_values.append(evaluate_labels(labels))
            NDCG_1_values.append(evaluate_labels(labels,1))
            NDCG_3_values.append(evaluate_labels(labels,3))
            NDCG_5_values.append(evaluate_labels(labels,5))
            NDCG_10_values.append(evaluate_labels(labels,10))

        #Normalize and update
        theta += theta_acc/n_queries
        theta=theta/np.linalg.norm(theta)
        W += W_acc/n_queries
        W = W/np.linalg.norm(W)

        #Evaluate the training
        print("Total Return(across all querys):",np.round(G_total,5))
        print("Avg Return(per query):",np.round(G_total/n_queries,5))
        print("Avg Reward(per step):",np.round(r_epoch_step_mean/n_queries,5))
            
        # After completion of all queries:Mean NDCGs
        NDCG_mean = np.mean(np.array(NDCG_values))
        NDCG_1_mean = np.mean(np.array(NDCG_1_values))
        NDCG_3_mean = np.mean(np.array(NDCG_3_values))
        NDCG_5_mean = np.mean(np.array(NDCG_5_values))
        NDCG_10_mean = np.mean(np.array(NDCG_10_values))

        #Updating results dictionary
        results['training'].append((NDCG_mean, NDCG_1_mean, NDCG_3_mean,NDCG_5_mean, NDCG_10_mean,np.round(r_epoch_step_mean/n_queries,7),np.round(G_total,5),np.round(G_total/n_queries,5)))

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

def Train_Full_Split_episodicupdate(data,n_features, theta,W,epochs,alpha_actor,alpha_critic,gamma,seed,trainprop):
    """Trains policy for given number of iterations using theta,W as starting point """

    gamma = gamma
    alpha_theta=alpha_actor
    alpha_w= alpha_critic
    # n_queries=100
    # len_episode=20

    #Import dataset and split into train and test sets
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

    #Results dictionary initialization
    results={'training':[],'test':[]}

    #Initial Values
    print("Initial Training Scores:")
    ndcg_values = Test_Queries(theta, data,train_queries_list,n_features)
    print(f"NDCG mean value = {ndcg_values[0]} NDCG@1 mean value = {ndcg_values[1]} NDCG@3 mean value = {ndcg_values[2]} NDCG@5 mean value = {ndcg_values[3]} NDCG@10 mean value = {ndcg_values[4]}")
    results['training'].append(tuple(ndcg_values))

    print("\nInitial Test Scores:")
    ndcg_values = Test_Queries(theta, data,test_queries_list,n_features)
    print(f"NDCG mean value = {ndcg_values[0]} NDCG@1 mean value = {ndcg_values[1]} NDCG@3 mean value = {ndcg_values[2]} NDCG@5 mean value = {ndcg_values[3]} NDCG@10 mean value = {ndcg_values[4]}")
    results['test'].append(tuple(ndcg_values))
    print()
    
    
    #Main training loop
    for i in range(1,epochs+1):
        #Monitor time taken
        e_start=time.time()

        # print("\n***********EPOCH: " + str(i) + " START***********\n")
        NDCG_values = []
        NDCG_1_values = []
        NDCG_3_values = []
        NDCG_5_values = []
        NDCG_10_values = []

        #Monitors
        G_total=0
        r_epoch_step_mean=0

        for q in train_queries_list:

            #initialize buffers
            theta_acc=np.zeros((n_features,1))
            W_acc=np.zeros((n_features,1))

            #Initialize state
            t=0
            docidsleft=list(data[q].keys())
            M=len(docidsleft)
            s=[t,docidsleft.copy()]
            
            #Monitors
            epi_step_r=0
            labels=[]

            #Take Action 
            a=sample_action(theta,s,data,q)
            labels.append(data[q][a][1])
            
            effective_length=M

            while(t!=M-1):
                #Get reward
                r=get_reward(a,t,data,q)
                G_total+=r
                epi_step_r+=r

                #Change state
                t=t+1
                docidsleft.pop(docidsleft.index(a))
                s_next=[t,docidsleft.copy()]
                
                #Take Action
                a_next=sample_action(theta,s_next,data,q)
                labels.append(data[q][a_next][1])
                
                #Find the scorefn of s,s_new
                grad_s_a=compute_scorefn(s,a,theta,data,q)
                grad_s_next_a_next=compute_scorefn(s_next,a_next,theta,data,q)
                
                #define the psi feature vector
                psi_s_a=grad_s_a
                psi_s_next_a_next=grad_s_next_a_next
                
                #Find the Q values
                q_s_a=compute_qvalue(W,psi_s_a)#Compute the q(s,a)
                q_s_next_a_next=compute_qvalue(W,psi_s_next_a_next)#Compute q(s_next,a_next)
                
                #updates
                theta_acc+=alpha_theta*grad_s_a*q_s_a #Policy Update
                td_error=r+gamma*q_s_next_a_next - q_s_a #td(0) error
                W_acc+= alpha_w*td_error*psi_s_a #Critic Update
                
                #Change state variables
                s=s_next
                a=a_next

            r_epoch_step_mean+=epi_step_r/effective_length
            
            #Training NDCGS for current query
            #print("\nlabels:",labels)
            NDCG_values.append(evaluate_labels(labels))
            NDCG_1_values.append(evaluate_labels(labels,1))
            NDCG_3_values.append(evaluate_labels(labels,3))
            NDCG_5_values.append(evaluate_labels(labels,5))
            NDCG_10_values.append(evaluate_labels(labels,10))

            #Update and normalize
            theta += theta_acc
            theta=theta/np.linalg.norm(theta)
            W += W_acc
            W = W/np.linalg.norm(W)

        #Evaluate the training
        print("Total Return(across all querys):",np.round(G_total,5))
        print("Avg Return(per query):",np.round(G_total/n_queries,5))
        print("Avg Reward(per step):",np.round(r_epoch_step_mean/n_queries,5))
            
        # After completion of all queries:Mean NDCGs
        NDCG_mean = np.mean(np.array(NDCG_values))
        NDCG_1_mean = np.mean(np.array(NDCG_1_values))
        NDCG_3_mean = np.mean(np.array(NDCG_3_values))
        NDCG_5_mean = np.mean(np.array(NDCG_5_values))
        NDCG_10_mean = np.mean(np.array(NDCG_10_values))

        #Updating results dictionary
        results['training'].append((NDCG_mean, NDCG_1_mean, NDCG_3_mean,NDCG_5_mean, NDCG_10_mean,np.round(r_epoch_step_mean/n_queries,7),np.round(G_total,5),np.round(G_total/n_queries,5)))

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