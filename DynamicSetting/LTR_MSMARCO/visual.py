# Librarys required
import tkinter
from tkinter import messagebox, IntVar, simpledialog
import webbrowser
import ltr
import random
import Agent
import torch
import numpy as np
import time
import rank as rk
import args

torch.manual_seed(6)
np.random.seed(6)
random.seed(6)


# To bind the url to the title
def open_url(urlStr):
    """utility function for hyperlinks functionality"""
    webbrowser.open(urlStr)


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
    # Current query string
    queryStr = ''
    # all displayed label's URL
    urlStr = []
    # all displayed label's title 
    titleStr = []
    # List of all query Ids
    topicList = []
    # Stored all queries strings in dict
    querystring = {}
    # query-doc for K documents shown to user {query: [doc]}
    query_doc_K_dict = {}
    # Indexing structure for dataset stored for fast access of data
    docoffset = {}
    # Current ID of the query
    topicId = ''
    # Current list of documents shown on screen(all pages)
    docLst = []
    # Current list of documents shown in the current page
    pageLst = []
    # GUI Label Ids stored for updation and print
    labelList = []
    # Dictionary of user feedback : {queryId: {docId: position}}
    userFeedbacks = {}
    # Stores a sample of episode by agent : {queryid: [state,action]}
    sample_episode = {}
    # Stores a list of actions (or ranking of documents) done by the agent: {queryId: [docIds]}
    action_list = {}
    # Dictionary of documents and their relevance scores
    doc_M_rs = {}
    # All the document IDs present in corpus (superset of M)
    df_lst = []
    # List of parameters for the agent
    W = []
    # To fetch the actual data from the title and url file opened
    f = open("msmarco-docs.tsv", encoding='utf8')


    # Initialize the window, radiobuttons, and the timestep. 
    # N denotes number of documents across the pages
    def __init__(self, window, alpha, beta, N):
        self.window = window
        self.radioselect = IntVar()

        # Loading model for doc2vec
        self.model = ltr.LoadModel("d2v_corpus_set_1m.model")
        self.t = 0
        self.N = N

        # storing retrieval parameters alpha and beta
        self.alpha = alpha
        self.beta = beta


    # Button Mapped function to move to the next page
    def nextPage(self, lbl_queryStr):
        if self.page >= 4:
            messagebox.showinfo('Error', 'You have reached the last page')
        else:
            lbl_queryStr.config(text="Current Query:  \"" + self.querystring[self.topicId] + "\"")
            self.page+=1
            self.pageLst = self.docLst[(self.page-1)*10:self.page*10]

            # Updating screen with new labels
            self.updateStrings(self.pageLst)
            self.updateLabels()


    # Button Mapped function to move to the previous page
    def prevPage(self, lbl_queryStr):
        if self.page <= 1:
            messagebox.showinfo('Error', 'You have reached the first page')
        else:
            lbl_queryStr.config(text="Current Query:  \"" + self.querystring[self.topicId] + "\"")
            self.page -= 1
            self.pageLst = self.docLst[(self.page-1)*10:self.page*10]

            # Updating screen with new labels
            self.updateStrings(self.pageLst)
            self.updateLabels()


    #Changes the top N results labels with contents
    def updateLabels(self):
        for i in range(10):
            self.labelList[i].config(text=self.titleStr[i])
            self.labelList[i].bind("<Button-1>", lambda e, url=self.urlStr[i]: open_url(url))


    # Updates the strings for the labels based on the document list passed
    def updateStrings(self, pageLst):
        """Updates the contents of the variables titleStr,urlStr from the given pageLst """
        self.urlStr = []
        self.titleStr = []
        iterDoc = 1

        for i in pageLst:
            docFetched = rk.getcontent(i, self.docoffset, self.f)

            # Use url as title when title is missing
            if len(docFetched[2]) <= 0 or docFetched[2] == '.':
                self.titleStr.append(str((self.page-1)*10 + iterDoc) + "." + docFetched[1])
            # print("Docid: "+i+" Content: "+docFetched[3][:40])
            else:
                self.titleStr.append(str((self.page-1)*10 + iterDoc) + "." + docFetched[2])

            self.urlStr.append(docFetched[1])
            iterDoc += 1


    # To display labels for the first time on the screen
    def printLabels(self):
        for i in range(10):
            label = tkinter.Label(self.window, text=self.titleStr[i], wraplength=550, justify="left",fg="blue", cursor="hand2")
            label.grid(row=i + 2, column=0, sticky='W')
            label.bind("<Button-1>", lambda e, url=self.urlStr[i]: open_url(url))
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
            messagebox.showinfo('popup', f'Recording feedback: {((self.page-1) * 10) + (indexSelected + 1)}')
            
            # Check if feedback to the document is already given
            if (1,self.topicId) not in self.userFeedbacks:
                self.userFeedbacks[(1,self.topicId)] = {}
            
            # Add feedback with value as position of the document
            self.userFeedbacks[(1,self.topicId)][self.pageLst[indexSelected]] = ((self.page-1) * 10) + (indexSelected + 1)

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
        userFeedbacks = ltr.generate_userFeedback(self.queryStr, list(self.docLst), number_users, self.model)

        # Check if feedback to the document is already given    
        if (0, self.topicId) not in self.userFeedbacks:
            self.userFeedbacks[(0, self.topicId)] = {}
        for item in list(userFeedbacks):
            self.userFeedbacks[(0, self.topicId)][item] = userFeedbacks[item]

        print("\nUser feedbacks recorded so far")
        print(self.userFeedbacks)


    # Function Mapping to the next query
    def nextQuery(self, lbl_queryStr):
        # Fetches the next query and updates the UI to display the corresponding results.
        print("\n-------GENERATING NEXT QUERY AND DOCUMENT LIST --------\n")

        try:
            topicInput = simpledialog.askstring("Input", "Enter the query ID to go to:", parent = self.window)
            if topicInput == None or topicInput == '':
                return
            else:
                self.queryStr = self.querystring[topicInput]
                self.topicId = topicInput

        except:
            print("Invalid input\n\n")
            messagebox.showinfo('Next Query', 'Failed to load query')
            return        

        # Retrieving N documents
        query_doc_M_retrieved = ltr.query_retrieval(self.queryStr, self.doc_M_rs, self.alpha, self.beta, self.model)
        self.docLst = list(query_doc_M_retrieved)[0:self.N]
        lbl_queryStr.config(text="Current Query:  \"" + self.querystring[self.topicId] + "\"")

        # Ranking based on policy 
        self.sample_episode[self.topicId], self.action_list[self.topicId] = Agent.rank(list(self.docLst), 
                                                                                            self.topicId, 
                                                                                            self.docoffset, 
                                                                                            self.model, self.W)
        
        # Update screen with new list
        messagebox.showinfo('Next Query', 'Next Query Loaded')
        self.page = 1
        self.query_doc_K_dict[self.topicId] = self.action_list[self.topicId]
        self.docLst = self.action_list[self.topicId]
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
        self.doc_M_rs = ltr.updateRelevanceScore(self.doc_M_rs, self.userFeedbacks, decay_value)

        ## Update the agents policy based on the relevance score --> rewards
        print("\n---Generating rewards based on the Relevance Score and updating the agent's policy---\n")
        self.W = Agent.update(self.W, self.userFeedbacks, self.doc_M_rs, self.query_doc_K_dict, self.sample_episode, 
                            self.docoffset, self.model, gamma, num_features, learning_rate)

        ## Add N more documents
        self.doc_M_rs = ltr.extendDocList(self.doc_M_rs, len_add_docs, self.model, self.df_list)

        ## Do sliding window on the M documents
        print("\n---Performing Sliding window on the documents---\n")
        self.doc_M_rs = ltr.slideWindow(self.doc_M_rs, window_size)

        ## Retrieve again for current query
        print("\n---Performing Retrieval based on Similarity and relevance scores---\n")
        query_doc_M_retrieved = ltr.query_retrieval(self.queryStr, self.doc_M_rs, self.alpha, self.beta, self.model)

        self.docLst = list(query_doc_M_retrieved)[0:self.N]

        ## Ranking based on updated weights 
        print("\n---Ranking based on Agent's policy---\n")
        self.sample_episode[self.topicId], self.action_list[self.topicId] = Agent.rank(list(self.docLst), 
                                                                                            self.topicId, 
                                                                                            self.docoffset, 
                                                                                            self.model, self.W)

        ## Update screen
        messagebox.showinfo('Update', f'Updated to time {self.t}')
        self.docLst = self.action_list[self.topicId]
        self.query_doc_K_dict[self.topicId] = self.docLst
        self.page = 1
        self.updateStrings(self.docLst)
        self.updateLabels()


        self.userFeedbacks = {}
        print(f"\nUpdated to time t = {self.t}\n")


def main():
    # Take input from user
    M, decay_value, len_add_docs, N, gamma, alpha, beta, num_features, learning_rate, window_size, corpus_size = args.fetchArgs()

    # Basic window settings
    window = tkinter.Tk()
    window.title("LTR_v6")
    window.columnconfigure(0, weight=1)
    window.columnconfigure(1, weight=1)
    window.columnconfigure(2, weight=1)
    window.columnconfigure(3, weight=1)
    window.geometry('800x450')
    buttonContainer = tkinter.Frame(window)
    buttonContainer.grid(column=0, row=13, columnspan=4)

    ts = time.time()
    visual_obj = VisualInterface(window, alpha, beta, N)
    print(f"Time taken to load model {time.time() - ts}")

    ## Initial run
    # Fetching the dataset, and the queries
    print("\n---Initializing time-step t = 0---\n")
    visual_obj.topicList, visual_obj.querystring, visual_obj.docoffset = rk.main()

    # Initializing each doc with relevance score of 1
    visual_obj.doc_M_rs, visual_obj.df_list = ltr.generate_M_docs(M, visual_obj.model, corpus_size)

    # Using the first query in the list
    visual_obj.topicId = visual_obj.topicList[0]
    visual_obj.queryStr = visual_obj.querystring[visual_obj.topicId]

    # Retrieving N documents for the query based on SS and RS
    query_doc_M_retrieved = ltr.query_retrieval(visual_obj.queryStr, visual_obj.doc_M_rs, alpha, beta, visual_obj.model)

    # Showing these documents to user
    visual_obj.docLst = list(query_doc_M_retrieved)[0:N]

    print("\n--- Completed Initialization ---\n")

    # Initial Ranking of docList

    # Initialize the weights
    visual_obj.W = Agent.initialize_weights(num_features)
    
    # Initial Ranking based on initialized weights 
    visual_obj.sample_episode[visual_obj.topicId], visual_obj.action_list[visual_obj.topicId] = Agent.rank(list(visual_obj.docLst), 
                                                                                                                visual_obj.topicId, 
                                                                                                                visual_obj.docoffset, 
                                                                                                                visual_obj.model, visual_obj.W)

    # Label Definitions
    lbl_queryStr = tkinter.Label(window, wraplength=750, text="Current Query:  \"" + visual_obj.queryStr + "\"")
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
