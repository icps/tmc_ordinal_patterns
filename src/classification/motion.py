import numpy as np
import pandas as pd


class MotionDataset:
    """ It calculates the point features chosen by the user.
    
    Parameters
    ----------
        no value
        
    Returns
    -------
        no value
    """   
        
    def _feature_extraction(self, feature):
        
        fmin  = np.min(feature)
        fmax  = np.max(feature)
        fmean = np.mean(feature)
        fstd  = np.std(feature)
        
        feature_list = [fmin, fmax, fmean, fstd]
        
        return feature_list
        
    
    
    def build_dataset(self, motion_features, path_transports):
        """ It gets the motion features and calculates the chosen point features.
        
        Parameters
        ----------
            folder_features : str
                absolute path where the motion feature data is
                
            motion_features : list of str
                motion feature names
            
            point_features : list of str
                point feature names, which will be extracted from the motion features
                
            transportation : list of str
                list of transportation mode name used to build the dataset
                
        Returns
        -------
            X : pandas dataframe
                dataframe of features 
                length: motion features * parameter
                
            y : pandas dataframe
                class labels
        """
        
        feature_df = pd.DataFrame()  
            
        for file in path_transports:

            df_motion = pd.read_csv(file, usecols = motion_features)
            
            calc_point_features = df_motion[motion_features].apply(self._feature_extraction, axis = 0).to_numpy()
            
            feature_aux = pd.DataFrame()
            
            for features in calc_point_features:
                
                df_features = pd.DataFrame([features])                
                feature_aux = pd.concat([feature_aux, df_features], axis = 1, ignore_index = True)
                
            feature_df = pd.concat([feature_df, feature_aux], axis = 0, ignore_index = True)
        
        return feature_df