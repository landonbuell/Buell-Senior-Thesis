"""
Landon Buell
PHYS 799
Instrument Classifier v0
12 June 2020
"""

            #### IMPORTS ####

import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf

import System_Utilities as sys_utils
import Feature_Utilities as feat_utils
import Plotting_Utilities as plot_utils
import Machine_Learning_Utilities as ML_utils
import Neural_Network_Utilities


"""
Mode_Utilities.py - 'Mode Utilities'
    Contains Definitions that are only called directly from MAIN script
    Functions are large & perform Groups of important operations
"""

            #### FUNCTION DEFINITIONS ####  

class Program_Mode:
    """
    Base Program mode object from which all programs objects inherit from
    --------------------------------
    FILEOBJS (iter) : List-like of File_object instances
    model_names (iter) : List-like of strings calling Network models by name
    n_classes (int) : number of discrete classes for models
    show_summary (bool) : If True, result are displayed to user
    group_size (int) : number of file samples in each design matrix
    --------------------------------
    Execute MAIN program in perscribed mode
    """
    def __init__(self,FILEOBJS,model_names,n_classes,show_summary=True,group_size=256):
        """ Inititialize Class Object Instance """
        self.FILEOBJS = FILEOBJS
        self.model_names = model_names
        self.n_classes = n_classes
        self.show_summary = show_summary    # show results of stages
        self.n_files = len(self.FILEOBJS)
        self.group_size = group_size

    def loop_counter(self,cntr,max,text):
        """ Print Loop Counter for User """
        print('\t\t('+str(cntr)+'/'+str(max)+')',text)
        return None

    def collect_features (self,fileobj):
        """ Collected Features from a Given .WAV file object"""
        fileobj = fileobj.read_audio()          # read raw .wav file
        # Time series feature object
        time_features = feat_utils.Time_Series_Features(fileobj.waveform,fileobj.rate)       
        # Frequency series feature object
        freq_features = feat_utils.Frequency_Series_Features(fileobj.waveform,
                                        fileobj.rate,time_features.frames)

        x1 = ML_utils.Feature_Array(fileobj.target)     # Structure holds MLP features
        x1 = x1.set_features(np.zeros(shape=(15,)))     # set features
        x2 = ML_utils.Feature_Array(fileobj.target)     # strucutre to hold Sxx features
        x2 = x2.set_features(freq_features.Sxx)         # set spectrogram
        x3 = ML_utils.Feature_Array(fileobj.target)     # structure to hold PSC features
        x3 = x3.set_features(np.zeros(shape=(2,2048)))  # set features

        return x1,x2,x3

    def construct_design_matrices (self,FILES):
        """ Collect Features from a subset File Objects """
        X1 = ML_utils.Design_Matrix(ndim=2,n_classes=self.n_classes)  # Design matrix for MLP
        X2 = ML_utils.Design_Matrix(ndim=4,n_classes=self.n_classes)  # Design matrix for Spectrogram
        X3 = ML_utils.Design_Matrix(ndim=4,n_classes=self.n_classes)  # Design matrix for Phase-Space

        for i,FILEOBJ in enumerate(FILES):
            self.loop_counter(i,len(FILES),FILEOBJ.filename)    # print messege
            x1,x2,x3 = self.collect_features(FILEOBJ)           # collect features
            X1.add_sample(x1)   # Add sample to MLP
            X2.add_sample(x2)   # Add sample to Sxx
            X3.add_sample(x3)   # add sample to Psc

        X1 = X1.shape_by_sample()
        X2 = X2.pad_2D(new_shape=Neural_Network_Utilities.spectrogram_shape)
        X3 = X3.pad_2D(new_shape=Neural_Network_Utilities.phasespace_shape)
        
        return [X1,X2,X3]

class Train_Mode (Program_Mode):
    """
    Run Program in 'Train Mode'
        Inherits from 'Program_Mode' parent class
    --------------------------------
    FILEOBJS (iter) : List-like of File_object instances
    model_names (iter) : List-like of strings calling Network models by name
    n_classes (int) : number of discrete classes for models
    show_summary (bool) : If True, result are displayed to user
    n_iters (int) : Indicats how may iterations to do over the full data 
    --------------------------------
    Creates Program Train Mode Object
    """
    def __init__(self,FILEOBJS,model_names,n_classes,show_summary=True,n_iters=2):
        """ Instantiate Class Method """
        super().__init__(FILEOBJS,model_names,show_summary)
        self.n_iters = n_iters

        # For each model store a list of history objs
        self.model_histories = {self.model_names[0]:[],
                                self.model_names[1]:[],
                                self.model_names[2]:[]}

    def __TRAIN__(self,Networks):
        """ Train Netorks on data from FILEOBJS """
        group_cntr = 0
        epcs = 4       # training epochs on subset
        for I in range (0,self.n_files,self.group_size):    # In a given group
            print("\t\t\tGroup Number:",group_cntr)
            FILES = self.FILEOBJS[I:I+self.group_size]      # subset of files
            Design_Matrices = super().construct_design_matrices(FILES)                                                   
            for dataset,model in zip(Design_Matrices,self.model_names):
                print("\t\t\tLoading & Fitting Model:",model)
                MODEL = Networks.__loadmodel__(model)   # Load network
                X = dataset.__get_X__()                 # Features
                Y = dataset.__get_Y__()                 # Labels          
                history = MODEL.fit(x=X,y=Y,batch_size=32,epochs=epcs,verbose=2,
                                    initial_epoch=(group_cntr*epcs))
                Networks.__savemodel__(MODEL)           # save model
                self.store_history(history,model)       # store data
            group_cntr += 1                     # incr coutner
        return self                             # self

    def store_history (self,history_object,model):
        """ Store Keras History Object in lists """
        assert model in self.model_names    # must be a known model
        self.model_histories[str(model)].append(history_object)
        return self

    def __call__(self,Networks):
        """ Call this Instance to Execute Training and Testing """
        print("\nBegining Training process....")
        for I in range (0,self.n_iters):
            print("\tIteration:",I)
            self.__TRAIN__(Networks)
            self.FILEOBJS = np.random.permutation(self.FILEOBJS)
        print("\tTraining Completed! =)")
        self.export_results()
        return self

    def aggregate_history (self):
        """ Aggregate history objects by model type & metric type """
        pass

    def export_results (self):
        """ Export Results of training models to a local path """
        pass

class Test_Mode (Program_Mode):
    """
    Run Program in 'Test Mode'
        Inherits from 'Program_Mode' parent class
    --------------------------------
    FILEOBJS (iter) : List-like of File_object instances
    model_names (iter) : List-like of strings calling Network models by name
    n_classes (int) : number of discrete classes for models
    show_summary (bool) : If True, result are displayed to user
    labels_present (bool) : If True, evaluation labels are given
    --------------------------------
    Creates Program Test Mode Object
    """
    def __init__(self, FILEOBJS,model_names,n_classes,show_summary=True,
                 labels_present=False,prediction_threshold=0.5):
        """ Initialize Class Object Instance """
        super().__init__(FILEOBJS,model_names)
        self.labels_present = labels_present        # labels for the trainign set?
        self.prediction_threshold = prediction_threshold
        if self.labels_present == True:     # we have labels!
            self.labels = np.array([])      # hold them in list!
            self.losses = { self.model_names[0]:np.array([]),
                            self.model_names[1]:np.array([]),
                            self.model_names[2]:np.array([])}

        # For each model store a list of history objs
        self.model_predictions = {  self.model_names[0]:np.array([]),
                                    self.model_names[1]:np.array([]),
                                    self.model_names[2]:np.array([])}

    def __TEST__(self,Networks):
        """ Test Netorks on data from FILEOBJS """
        group_cntr = 0
        for I in range (0,self.n_files,self.group_size):# In a given group
            print("\tGroup Number:",group_cntr)
            FILES = self.FILEOBJS[I:I+self.group_size]  # subset of files
            Design_Matrices = super().construct_design_matrices(FILES)
            epcs = 16                                   # trainign epochs
            for dataset,model in zip(Design_Matrices,self.model_names):
                print("\t\t\tLoading & Testing Model:",model)
                MODEL = Networks.__loadmodel__(model)   # Load network
                X = dataset.__get_X__()                 # Features
                self.__predict__(MODEL,X)           # run predictions
                if self.labels_present == True:         # if we have labels               
                    Y = dataset.__get_Y__()             # Labels
                    self.__evaluate__(MODEL,X,Y)        # run evaluation
                                   
                Networks.__savemodel__(MODEL)           # save model              
            group_cntr += 1                 # incr counter
        return self                         # return self       

    def __predict__(self,model,X):
        """ Predict Class output from unlabeled sample features """
        y_pred = model.predict(x=X,verbose=0)   # get network predicitons
        # store predictions
        self.model_predictions[str(model.name)] = \
            np.append(self.model_predictions[str(model.name)],y_pred)
        return self                             # return itself

    def __evaluate__(self,model,X,Y):
        """ Compute Loss value and metrics from labeled sample features """
        result = model.evaluate(x=X,y=Y,verbose=0,return_dict=False)    # build-in evaluation
        return self

    def __call__(self,Networks):
        """ Call this Instance to Execute Training and Testing """
        print("\nBegining Testing Process...")
        self.__TEST__(Networks)
        print("\tTesting Completed! =)")

    def export_results (self):
        """ Export Results of training models to a local path """
        pass

class TrainTest_Mode (Program_Mode):
    """
    Run Program in 'Train_Mode' and 'Test Mode' sequentially
        Inherits from 'Train_Mode' and 'Test Mode' parent classes
    --------------------------------
    FILEOBJS (iter) : List-like of File_object instances
    model_names (iter) : List-like of strings calling Network models by name
    n_classes (int) : number of discrete classes for models
    show_summary (bool) : If True, result are displayed to user
    labels_present (bool) : If True, evaluation labels are given
    testsize (float) : Value on interval (0,1) indicate fraction of data to test with
    --------------------------------
    Creates Program Test Mode Object
    """
    def __init__(self, FILEOBJS,model_names,n_classes,show_summary=True   ,
                 labels_present=True,testsize=0.1):
        """ Initialize Class oject instance """
        super().__init__(FILEOBJS,model_names,show_summary)
        self.testsize=testsize          # train/test size
        self.labels_present = labels_present    # labels?
        self.Split_Objs()                   # split objs
                  
    def Split_Objs (self):
        """ Split objects into training.testing subsets """
        train,test = train_test_split(self.FILEOBJS,test_size=self.testsize)
        delattr(self,'FILEOBJS')        # delete attrb
        self.TRAIN_FILEOBJS = train     # set attrb
        self.n_train_files = len(self.TRAIN_FILEOBJS)
        self.TEST_FILEOBJS = test       # set attrbs
        self.n_test_files = len(self.TEST_FILEOBJS)
        return self                     # return self

    def __call__(self,Networks):
        """ Call this Instance to Execute Training and Testing """

        # Run Training Mode
        Training = Train_Mode(self.TRAIN_FILEOBJS,self.model_names,
                              self.n_classes,self.show_summary)
        Training.__call__(Networks)

        # Run Testing Mode
        Testing = Test_Mode(self.TEST_FILEOBJS,self.model_names,
                            self.n_classes,self.show_summary,self,labels_present)
        Testing.__call__(Networks)
            
        return self


        