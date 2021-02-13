#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 17:57:09 2020

@author: isadora
"""

import pandas as pd
from os.path import join

from multiprocessing import Pool

from src.op_transformation.ordinal_patterns import OPTransformation
from src.utilities.utils import get_files, create_folder


class OPTransformationExtraction:
    """ From each data segment, calculates OP Transformation.
    It uses multiprocessing to accelerate.
    """

        
    def _save_op_transformation(self, df_op, path_to_save, file_name):
        """ Saves the OP transformation.
        
        Parameters
        ----------
            df_op : pandas dataframe
                the transformations to save
                
            path_to_save : str
                absolute path where to save the df
                
            file_name : str
                name to save data
                                
        Returns
        -------
            no value
        """
        
        ## save data
        path_to_save = join(path_to_save, file_name)
        
        create_folder(path_to_save)
                    
        df_op.to_csv(path_to_save, sep = ",", header = True, index = None)


    def _op_pooling(self, params):
        """ Auxiliar function for multiprocessing. 
        
        Parameters
        ----------
            params : tuple (pandas dataframe, int, int, int)
                time series to apply transformation and OP parameters
                
        Returns
        -------
            results : tuple
                type depends on the function it calls
                if calling op.*features, it is a tuple of float
                if calling op.*distribution, it is a tuple of lists
        """
        
        # for the multiprocessing
        op_features, data, D, tau = params

        op                        = OPTransformation()

        result                    = op.get_features(op_features, data, D, tau, 
                                                    numred = False, prob = False)
    
        return result
    
    
    def _op_multithread(self, X, D, tau, op_features):
        """ Auxiliar function for multiprocessing. 
        
        Parameters
        ----------
            X : pandas dataframe
                time series to apply transformation
                (a time series is a single row of a dataset,
                hence, a dataset is composed of several time series)
                
            D : int
                the embedding dimension D
                
            tau : int
                the embedding delay tau
                
            op_features : list of str
                which features to extract from OP and OPTN transformation
                            
        Returns
        -------
            op_data : pandas dataframe
                dataframe containing the OP transformation of time series
        """
        
        dataset = [(op_features, row, D, tau) for index, row in X.iterrows()]
                
        p       = Pool()
        op_data = p.map(self._op_pooling, dataset)
        p.close()
        
        return op_data
    
    
    def get_transformation(self, parameters, motion_features, op_features, 
                           transportation, folder_features, folder_op):
        """ From each motion feature, calculates OP transformation,
        saving it to the chosen folder.
        The feature in dataset is a column, but our OP functions are
        implemented to rows, so we transpose each feature
        
        Parameters
        ----------
            parameters : list of lists (int, int)
                OP parameters (D, tau) we want to extract 
                
            motion_features : list of str
                list of motion features we want to transform
                
            op_features : list of str
                which features to extract from OP and OPTN transformation
                
            transportation : list of str
                list of transportation modes which will be transformed
                
             folder_features : str
                absolute path where the motion features that will be transformed are
                
            folder_op : str
                the folder where to save the segments  
                            
        Returns
        -------
            no value
        """
        
        for D, tau in parameters:
                
            print('\n for D = {} and tau = {} \n'.format(D, tau))
            
            for motion_feature in motion_features:
                                            
                for transport in transportation:
                    
                    query = transport + "*.csv"
                    user_files = get_files(folder_features, query, True)
                    
                    # dict of lists
                    keys  = op_features
                    df_op = {k: [] for k in keys}
                    
                    print("Motion Feature: {} and transportation: {}".format(motion_feature, transport))    
                    
                    for file in user_files:
                        df = pd.read_csv(file, sep = ",", header = 0, usecols = [motion_feature])
                        
                        df.dropna(inplace = True)
                                                                    
                        # we have columns, but the op function gets a row
                        df_transposed = df.T # transposed                                                                        
                        
                        # we put [0] because we have 
                        # pe, sc, fi = [(2.31, 0.23, 0.045)] (does not work)
                        # and we want pe, sc, fi = (2.31, 0.23, 0.045)
                        #
                        feature_list = self._op_multithread(df_transposed, D, tau, op_features)[0]
                                                
                        for op, feat in zip(op_features, feature_list):
                            df_op[op].append(feat)
    
                    df_op = pd.DataFrame.from_dict(df_op)
                            
                    ## save data
                    op_values = "_D" + str(D) + "_t" +  str(tau)
                    file_name = "op_" + transport + "_" + motion_feature + op_values + ".csv"
                    
                    self._save_op_transformation(df_op, folder_op, file_name)

                
                
                
