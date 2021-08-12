import numpy as np
import pandas as pd
from os.path import join
from datetime import datetime
from itertools import product

from src.utilities.utils import create_folder, get_files
from src.classification.motion import MotionDataset
from src.classification.model_classification import Classification

np.random.seed(321)

class BaselineClassification:
    """ It classifies the motion features, extracting point features from it,
    so we can use these classification metrics as baseline.
    
    Parameters
    ----------
        dataset : str
            which dataset we are working on ("geolife" or "shl" so far)
            
        folder_features : str
            absolute path where the motion feature data is
            
        comparison : list of str
            specific to GeoLife, which transportation mode set we want to classify,
            based on previous works
            
        motion_features : list of str
            motion features names, which will be used to classify
        
        point_features : list of str
            point features names, which will be extracted from the motion features
            
        folder_baseline : str
            absolute path where to save the results
            
        model : sklearn classifier
            supervised model that will classify
    """
    
    def __init__(self, folder_features, motion_features, folder_baseline, model, n_samples):
        self.folder_features = folder_features
        self.motion_features = motion_features
        self.folder_baseline = folder_baseline
        self.model           = model
        self.n_samples       = n_samples
        
        filename = join(folder_baseline + "_" + str(datetime.now()), "")
        self.folder_baseline = filename  
        
        
    def _get_data(self, transportation):
        """ Reads the motion files and organized them in a
        single dataset: X for features and y for labels.
        
        Parameters
        ----------
            transportation : list of str
                list of transportation mode name used to classification
                
        Returns
        -------
            X : pandas dataframe
                dataframe of features 
                length: motion features * parameter
                
            y : pandas dataframe
                class labels
        """
        
        X = pd.DataFrame()
        y = pd.DataFrame()
        
        motion = MotionDataset()
            
        for transport in transportation:
    
            file_name      = transport + "*" + ".csv"
            path_transport = get_files(self.folder_features, file_name, True)
            
            feature_df     = motion.build_dataset(self.motion_features, path_transport)
            
            feature_df = feature_df[:self.n_samples]
            
            concat1 = [X, feature_df]
            X       = pd.concat(concat1, axis = 0, ignore_index = True)
            
            # labels
            motion_class = pd.DataFrame([transport] * len(feature_df))
            concat2      = [y, motion_class]
            y            = pd.concat(concat2, axis = 0, ignore_index = True)
                   
        print("#### Motion features size: {}".format(len(X.columns)))
        
        print("size", len(X), len(y))
              
        return X, y
        

    def _build_dataset(self, transports):
        """ Only for GeoLife dataset. 
        It build the dataset to classify, getting the point features
        (extracted from the motion features) and class labels. 
        The transport for the dataset are the set used on previous works, 
        so we can easily compare our results with them.
        
        Parameters
        ----------
            study : str
                name of previous work
                can be: "zheng", "dabiri", "xiao", or "jiang"
                
        Returns
        -------
            X : pandas dataframe
                dataframe of features 
                length: point features * motion features
                
            y : pandas dataframe
                class labels        
        """
        
        X, y       = self._get_data(transports)
            
        if "car" in transports or "taxi" in transports:
            y          = y.replace(to_replace = "car", value = "driving")
            y          = y.replace(to_replace = "taxi", value = "driving")
        
        
        print("TRANSPORTATION MODE: ", transports)
        
#         self._save_dataset(X, y)
        
        return X, y  
    
    
    def _save_dataset(self, X, y):
        """ It saves the dataset that will be used to classify 
        (helpful for plotting).
        
        Parameters
        ----------
            X : pandas dataframe
                dataframe of features 
                length: point features * motion features
                
            y : pandas dataframe
                class labels
                
            study : str
                name of previous work
                can be: "zheng", "dabiri", "xiao", or "jiang"
        """
        
        # Save dataset
        create_folder(self.folder_baseline)
        
        point_features = ["mean", "std"]#["min", "max", "mean", "std"]
        
        colnames     = list(product(self.motion_features, point_features))
        colnames     = list(map("_".join, colnames))
        colnames     = ["classes"] + colnames
        
        data         = pd.concat([y, X], axis = 1, ignore_index = True)
        data.columns = colnames
        filename     = self.folder_baseline + "DATASET_baseline.csv"
        
        data.to_csv(filename, index = False)
    

    def _save_results(self, df, file_name):
        """ It saves the classification results (metrics and confusion matrices)
        in the self.folder_baseline (it will create the folder, if it doesn't exist).
        
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
        
        create_folder(self.folder_baseline)
        
        path_to_save = join(self.folder_baseline, file_name)
        
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
            no value
                
        Returns
        -------
            no value
        """

            
        print("--- Saving at {}".format(self.folder_baseline))

        print("FEATURES: {}".format(str(self.motion_features)))

        X, y        = self._build_dataset(transports)

        clf         = Classification()
        standardize = True                 
        df, cm      = clf.classification(X, y, self.model, standardize, n_folds)

        # save data
        file_name = "METRICS_baseline.csv"
        self._save_results(df, file_name)

        file_name = "ConfusionMatrices_baseline.txt"
        self._save_results(cm, file_name)

        print("\n-------------------------------------------------")
        print("-------------------------------------------------\n")