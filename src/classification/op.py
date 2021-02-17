#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 23:12:46 2019

@author: isadora
"""

import numpy as np
import pandas as pd
from os.path import join
from datetime import datetime
from itertools import product

from src.utilities.utils import create_folder, get_files
from src.classification.model_classification import Classification


np.random.seed(321)


class OPClassification:
    """ It classifies motion (extracting point features from it) and OP features.
    
    Parameters
    ----------
        op_values : nested lists
            OP Transformation parameters (D, tau) used to classify
            ex: [[3, 1], [3, 1]]
                    
        motion_features : list of str
            motion feature names
            
        folder_op : str
            absolute path where the op features are
        
        folder_features : str
            absolute path where the motion feature data is
            
        folder_classification : str
            absolute path where to save the classification results
            (we add the time to each data, so the results won't overwrite)
            
        model : sklearn-based classifier
            supervised model that will classify
    """
    

    def __init__(self, op_values, motion_features, 
                 folder_op, folder_features, 
                 folder_classification, model, op_features):
        
        self.op_values       = op_values
        self.motion_features = motion_features
        self.folder_op       = folder_op
        self.folder_features = folder_features
        self.model           = model
        self.op_features     = op_features
                
        filename = join(folder_classification + "_" + str(datetime.now()), "")
        self.folder_classification = filename       
    
    
    def _get_data(self, transportation, parameter):
        """ Reads the OP Transformation files and organized them in a
        single dataset: X for features and y for labels.
        
        Parameters
        ----------
            transportation : list of str
                list of transportation mode name used to classification
                
            parameter : list of int
                list of OP parameters: D and tau
                
        Returns
        -------
            X : pandas dataframe
                dataframe of features 
                
            y : pandas dataframe
                class labels
        """
        
        D, tau        = parameter
        op_values     = "_D" + str(D) + "_t" + str(tau) + ".csv"
        features_name = [m + op_values for m in self.motion_features]
        # ex: 'distance_D3_t1.csv'

        X = pd.DataFrame()
        y = pd.DataFrame()    
        
        print("### OP features: {}".format(self.op_features))
        
        for transport in transportation:
            
            # path to op transformation files
            # ex: query      = 'op_bus_distance_D3_t1.csv'
            # ex: op_files = 'db/GeoLife/op_features/op_bus_distance_D3_t1.csv'
            file_name  = "op_" + transport + "_"
            query      = [file_name + f for f in features_name] 
            op_files   = [self.folder_op + q for q in query]
                                    
            df_transport_op = pd.DataFrame()            
            
            for file in op_files:      
                op_csv = pd.read_csv(file, usecols = self.op_features)
                op_csv = op_csv[self.op_features] # to assure order

                # axis = 1 is by column, axis = 0 is by rows
                concat           = [df_transport_op, op_csv]
                df_transport_op  = pd.concat(concat, axis = 1, ignore_index = True)
                df_transport_op  = df_transport_op.dropna()
                
            # features
            concat1    = [X, df_transport_op]
            X          = pd.concat(concat1, axis = 0, ignore_index = True)
       
            # labels
            op_class   = pd.DataFrame([transport] * len(df_transport_op))
            concat2    = [y, op_class]
            y          = pd.concat(concat2, axis = 0, ignore_index = True) 
                
            
        n = len(X.columns)
        print("#### OP features size: {}".format(n))
        
        return X, y
        

    def _build_dataset(self, parameter, transports):
        """ Only for GeoLife dataset. Get the transportation mode set to classify
        based on previous works (helps comparing the results)
        
        Parameters
        ----------
            parameter : list of int
                list of OP parameters: D and tau
                
        Returns
        -------
            X : pandas dataframe
                dataframe of features
                length: Information Theory features * motion features * parameter
                
            y : pandas dataframe
                class labels        
        """

            
#         transports = ["bus", "car", "taxi", "walk", "bike"]
        
        X, y       = self._get_data(transports, parameter)
            
        if "car" in transports or "taxi" in transports:
            y          = y.replace(to_replace = "car", value = "driving")
            y          = y.replace(to_replace = "taxi", value = "driving")
        
        
        print("TRANSPORTATION MODE: ", transports)
        
        self._save_dataset(X, y, parameter)
        
        return X, y   
    
    
    def _colnames(self, features):
        
        colnames = list(product(self.motion_features, features))
        colnames = list(map('_'.join, colnames))
        
        return colnames
        
    
    def _save_dataset(self, X, y, parameter):
        """ It saves the dataset that will be used to classify 
        (helpful for plotting).
        
        Parameters
        ----------
            X : pandas dataframe
                dataframe of features 
                
            y : pandas dataframe
                class labels
        """
        
        # Save dataset
        create_folder(self.folder_classification)
        
        ## get feature name        
        colnames_op  = self._colnames(self.op_features)
        
        colnames     = ["classes"] + colnames_op
        name         = "_op_" + str(parameter) + ".csv"
            
        data         = pd.concat([y, X], axis = 1, ignore_index = True)
        data.columns = colnames
        filename     = self.folder_classification + "DATASET_" + name
        
        data.to_csv(filename, index = False)               
      

    def _save_results(self, df, file_name):
        """ It saves the classification results (metrics and confusion matrices)
        in the self.folder_classification (it will create the folder, if it doesn't exist).
        
        Parameters
        ----------
            df : pandas dataframe
                the results to save
                
            file_name : str
                the name to save the result
                                
        Returns
        -------
            no value
        """
                
        create_folder(self.folder_classification)
        
        path_to_save = join(self.folder_classification, file_name)
        
        try:
            df.to_csv(path_to_save, sep = ",")
            
        except: # for confusion matrices
            
            df = np.array(df)
            with open(path_to_save, 'w') as outfile:
                for enum, data_slice in enumerate(df):
                    
                    if enum != 10:
                        # Writing out a break to indicate different slices...
                        outfile.write('# Confusion Matrix for CV{}\n'.format(enum+1))
                    else:
                        outfile.write('# Total Confusion Matrix\n')

                    np.savetxt(outfile, data_slice, fmt='%d')
            
                    
        
                    
    def classification(self, n_folds, transports):
        """ It calls the classification pipeline: join the dataset and classify it.
        Also, it saves the classification results in the chosen folder.
        
        Parameters
        ----------
            n_folds : int
                how many folds to divide data to cross-validation
                
        Returns
        -------
            no value
        """

        
        print("--- Saving at {}".format(self.folder_classification))        

        for parameter in self.op_values:

            print("OP PARAMETERS: {}".format(str(parameter)))
            print("FEATURES: {}".format(str(self.motion_features)))
                                    
            X, y        = self._build_dataset(parameter, transports)
            
            clf         = Classification()
            standardize = True
            df, cm      = clf.classification(X, y, self.model, standardize, n_folds)
            
            # save data
            filename = "METRICS_" + str(parameter) + ".csv"
            self._save_results(df, filename)

            filename = "ConfusionMatrices_" + str(parameter) + ".txt"
            self._save_results(cm, filename)
            
            print("\n-------------------------------------------------")
            print("-------------------------------------------------\n")
