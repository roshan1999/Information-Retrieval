# Libraries required
import tkinter
from tkinter import messagebox, IntVar, simpledialog
import ltr
import random
import PPO_Agent as Agent
import PPO
import torch
import numpy as np
import time
from Environment import Dataset
import args
torch.manual_seed(6)
np.random.seed(6)
random.seed(6)


"""
    class: VisualInterface: 
    Contains all the function definition and necessary variables for the 
    interaction between the GUI and the system.
"""
class VisualInterface:
    # Displays top N results
    N = 40
    page = 1

    ## variables to store required entities
    # title of the document
    titleStr = [] 
    # List of all query Ids
    topicList = [] 
    # query-doc for K documents shown to user {query: [doc]}
    query_doc_K_dict = {} 
    topicId = ''
    # K list of documents shown to user for the query
    docLst = [] 
    # List of documents in the current page
    pageLst = [] 
    # Stored list of GUI label ID
    labelList = [] 
    # Dictionary of user feedback : {(0/1, queryId): {docId: position}}
    userFeedbacks = {} 
    # Stores a sample of episode by agent : {queryid: [state,action]}
    sample_episode = {} 
    # Stores a list of actions (or ranking of documents) done by the agent: {queryId: [docIds]}
    action_list = {} 
    # Dictionary of documents and their relevance scores
    doc_M_rs = {} 
    # PPO Agent
    ppo_agent = [] 


    # Initialize the window, radiobuttons, and the timestep. 
    # N denotes number of documents across the pages
    def __init__(self, window, N, agent):
        self.window = window
        self.radioselect = IntVar()
        self.t = 0
        self.N = N
        self.agent = agent


    # Button Mapped function to move to the next page
    def nextPage(self, lbl_queryStr):
        if self.page >= 4:
            messagebox.showinfo('Error', 'You have reached the last page')
        else:
            lbl_queryStr.config(text="Current Query:  \"" + str(self.topicId) + "\"")
            self.page+=1
            self.pageLst = self.docLst[(self.page-1)*10:self.page*10]

            # Updating the screen with the new labels
            self.updateStrings(self.pageLst)
            self.updateLabels()


    # Button mapped function to move to the prev page
    def prevPage(self, lbl_queryStr):
        if self.page <= 1:
            messagebox.showinfo('Error', 'You have reached the first page')
        else:
            lbl_queryStr.config(text="Current Query:  \"" + str(self.topicId) + "\"")
            self.page -= 1
            self.pageLst = self.docLst[(self.page-1)*10:self.page*10]

            # Updating the screen with the new labels
            self.updateStrings(self.pageLst)
            self.updateLabels()


    # Changes the top N results labels with contents
    def updateLabels(self):
        for i in range(10):
            self.labelList[i].config(text=self.titleStr[i])


    # Updates the strings for the labels based on the document list passed
    def updateStrings(self, pageLst):
        self.titleStr = []
        iterDoc = 1

        for i in pageLst:
            self.titleStr.append(str((self.page-1)*10 + iterDoc) + "." + i)
            iterDoc += 1 

        # If all 10 strings not present, then replace with '-'
        while(iterDoc<11):
            self.titleStr.append(str((self.page-1)*10 + iterDoc) + "." + '-')
            iterDoc += 1


    # To display labels for the first time on the screen
    def printLabels(self):
        for i in range(10):
            label = tkinter.Label(self.window, text=self.titleStr[i], wraplength=550, justify="left",fg="blue", cursor="hand2")
            label.grid(row=i + 2, column=0, sticky='W')
            self.labelList.append(label)


    # Function mapping to record manual feedback, Here 1 = manual feedback
    def record_manualFeedback(self):
        """ Event Handler for the event:Optimize """

        print("\n------- RECORDING USER FEEDBACKS ------\n")
        messagebox.showinfo('popup', 'Storing manual feedback....')

        # index of selected query
        indexSelected = self.radioselect.get() - 1

        # If no relevant query selected by user
        if indexSelected == -1:
            messagebox.askokcancel("Error", "You have not selected any link, Please select a link")
        else:
            feedback = ((self.page-1) * 10) + (indexSelected + 1)
            messagebox.showinfo('popup', f'Recording feedback: {feedback}')

            # Check if feedback to the document is already given
            if (1,self.topicId) not in self.userFeedbacks:
                self.userFeedbacks[(1, self.topicId)] = {} # 1 indicates manual feedback

            # Add feedback with value as position of the document
            self.userFeedbacks[(1, self.topicId)][self.pageLst[indexSelected]] = feedback

        print("User feedbacks recorded so far: ")
        print(self.userFeedbacks)
        print()
        self.radioselect.set(0)

    
    # Function mapping to record automatic feedback, Here 0 = automatic feedback
    def generate_userFeedback(self):

        print("\n-------GENERATING USER FEEDBACKS AUTOMATICALLY --------\n")
        number_users = 0

        # Exception Testing
        try:
            number_users = int(simpledialog.askstring("Input", "How many user feedbacks to generate at once?", parent = self.window))
        except:
            print("Invalid input, taking number of user feedbacks as 0\n\n")
            
        if number_users == 0:
            print("No feedback generated")
            return

        # Generate automatic user feedbacks based on relevance labels available
        userFeedbacks = ltr.generate_userFeedback(self.topicId, self.docLst, number_users, self.data)

        # Check if feedback to the document is already given    
        if (0,self.topicId) not in self.userFeedbacks:
            self.userFeedbacks[(0, self.topicId)] = {} # 0 indicates simulated feedback
        for item in list(userFeedbacks):
            self.userFeedbacks[(0, self.topicId)][item] = userFeedbacks[item]

        print("\nUser feedbacks recorded so far")
        print(self.userFeedbacks)


    # Function Mapping to the next query
    def nextQuery(self, lbl_queryStr):

        # Fetches the next query and updates the UI to display the corresponding results.
        print("\n-------GENERATING NEXT QUERY AND DOCUMENT LIST --------\n")
        
        try:
            topicInput = int(simpledialog.askstring("Input", "Enter the query ID to go to:", parent = self.window))

            # Retrieving N documents
            curr_doc_list = ltr.retrieve(topicInput, self.doc_M_rs, self.data, self.N)
            self.topicId = topicInput

        except:
            print("Invalid input, Please enter a valid query ID \n\n")
            messagebox.showinfo('Next Query', 'Failed to load query')
            return
            
        lbl_queryStr.config(text="Current Query:  \"" + str(self.topicId) + "\"")

        # Ranking based on policy 
        self.sample_episode[self.topicId], self.action_list[self.topicId] = self.agent.rank(curr_doc_list, self.ppo_agent, self.data)

        messagebox.showinfo('Next Query', f'Query {self.topicId} Loaded')
        
        # Update screen with new list
        self.page = 1
        self.docLst = self.action_list[self.topicId]
        self.query_doc_K_dict[self.topicId] = self.docLst
        self.pageLst = self.docLst[(self.page-1)*10:self.page*10]
        self.updateStrings(self.pageLst)
        self.updateLabels()
        


    # Function mapping to update to next timestep
    def update(self, decay_value, len_add_docs, gamma, num_features, learning_rate, window_size):

        print("\n---------UPDATING---------\n")
        print("\nThe Stored user feedbacks are: \n")
        print(self.userFeedbacks)
        print(f"\nUpdating from time t = {self.t} to t = {self.t+1}\n")
        self.t += 1

        ## Update the relevance scores based on the user-feedbacks
        print("\n---Updating relevance scores of the document---\n")
        self.doc_M_rs = ltr.updateRelevanceScore(self.doc_M_rs, self.userFeedbacks, decay_value, self.data)

        ## Update the agents policy based on the relevance score --> rewards
        print("\n---Generating rewards based on the Relevance Score and updating the agent's policy---\n")
        self.agent.update(self.ppo_agent, self.userFeedbacks, self.doc_M_rs, self.query_doc_K_dict,self.N)

        ## Add N more documents
        self.doc_M_rs = ltr.extendDocList(self.doc_M_rs, len_add_docs, self.data)

        ## Do sliding window on the M documents
        print("\n---Performing Sliding window on the documents---\n")
        self.doc_M_rs = ltr.slideWindow(self.doc_M_rs, window_size)

        # Retrieving N documents for the query
        curr_doc_list = ltr.retrieve(self.topicId, self.doc_M_rs, self.data, self.N)
        
        ## Ranking based on updated weights 
        print("\n---Ranking based on Agent's policy---\n")
        self.sample_episode[self.topicId], self.action_list[self.topicId] = self.agent.rank(curr_doc_list, self.ppo_agent, self.data)

        ## Update screen
        messagebox.showinfo('Update', f'Updated to time {self.t}')
        self.page = 1
        self.docLst = self.action_list[self.topicId]
        self.query_doc_K_dict[self.topicId] = self.docLst
        self.pageLst = self.docLst[(self.page-1)*10:self.page*10]
        self.updateStrings(self.pageLst)
        self.updateLabels()


        self.userFeedbacks = {}
        print(f"\nUpdated to time t = {self.t}\n")


def main():
    ## Take input from user
    M, decay_value, len_add_docs, N, gamma, num_features, learning_rate, window_size, fname = args.fetchArgs()

    ## Basic window settings
    window = tkinter.Tk()
    window.title("LTR_DS")
    window.columnconfigure(0, weight=1)
    window.columnconfigure(1, weight=1)
    window.columnconfigure(2, weight=1)
    window.columnconfigure(3, weight=1)
    window.geometry('800x450')
    buttonContainer = tkinter.Frame(window)
    buttonContainer.grid(column=0, row=13, columnspan=4)
    agent = Agent.PPORank()
    visual_obj = VisualInterface(window, N, agent)

    ### Initial run
    ## Fetching the dataset, and the queries
    print("\n---Initializing time-step t = 0---\n")
    visual_obj.data = Dataset(fname)
    visual_obj.topicList = visual_obj.data.getTrain() # [query]
    visual_obj.doc_M_rs = ltr.generateFromLst(visual_obj.data, M)

    # Using the first query in the list
    visual_obj.topicId = visual_obj.topicList[0]

    # Retrieving N documents for the query
    curr_doc_list = ltr.retrieve(visual_obj.topicId, visual_obj.doc_M_rs, visual_obj.data, N)

    print("\n--- Completed Initialization ---\n")

    # initialize a PPO agent
    lr_actor =0.001
    lr_critic = 0.001
    gamma = 1
    K_epochs = 3
    eps_clip = 0.2
    visual_obj.ppo_agent = PPO.PPO(num_features, 1, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    
    # Initial Ranking based on initialized weights 
    visual_obj.sample_episode[visual_obj.topicId], visual_obj.action_list[visual_obj.topicId] = visual_obj.agent.rank(curr_doc_list, visual_obj.ppo_agent,visual_obj.data)

    # Label Definitions
    lbl_queryStr = tkinter.Label(window, wraplength=750, text="Current Query:  \"" + str(visual_obj.topicId) + "\"")
    lbl_queryStr.grid(column=0, row=0, sticky="W", columnspan=4)
    tkinter.Label(window, text="Top 10 results:").grid(row=1, sticky="W")

    # Initial updates
    visual_obj.docLst = visual_obj.action_list[visual_obj.topicId]
    visual_obj.query_doc_K_dict[visual_obj.topicId] = visual_obj.docLst
    visual_obj.pageLst = visual_obj.docLst[(visual_obj.page-1)*10:visual_obj.page*10]
    visual_obj.updateStrings(visual_obj.pageLst)
    visual_obj.printLabels()

    # Button Definitions
    for i in range(0, 10):
        btn_feed = tkinter.Radiobutton(window, text="Relevant", variable=visual_obj.radioselect, value=i + 1)
        btn_feed.grid(row=i + 2, column=1, columnspan=2, sticky="W", padx=15)
    
    btn_pgNext = tkinter.Button(buttonContainer, text="Next Page", command = lambda: visual_obj.nextPage(lbl_queryStr))
    btn_pgNext.grid(row=0, column=2, pady=15)
    
    btn_pgPrev = tkinter.Button(buttonContainer, text="Previous Page", command = lambda: visual_obj.prevPage(lbl_queryStr))
    btn_pgPrev.grid(row=0, column=1, pady=15)
    
    btn_Optimize = tkinter.Button(buttonContainer, text="Record User Feedback", command=visual_obj.record_manualFeedback)
    btn_Optimize.grid(row=1, column=0, padx=8)

    btn_Train = tkinter.Button(buttonContainer, text="Generate User feedbacks",
                               command=lambda: visual_obj.generate_userFeedback())
    btn_Train.grid(row=1, column=1, padx=8)

    btn_Train = tkinter.Button(buttonContainer, text="Update to Next Time-Step",
                               command=lambda: visual_obj.update(decay_value, len_add_docs, gamma, num_features, learning_rate, window_size))
    btn_Train.grid(row=1, column=2, padx=8)

    btn_queryNext = tkinter.Button(buttonContainer, text="Next Query",
                                   command=lambda: visual_obj.nextQuery(lbl_queryStr))
    btn_queryNext.grid(row=1, column=3, padx=8)

    window.mainloop()


if __name__ == "__main__":
    main()
