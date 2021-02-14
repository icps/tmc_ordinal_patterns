# -*- coding: utf-8 -*-


from itertools import chain 
from os.path import split, basename, dirname, join
from src.utilities.utils import get_files, create_folder
from src.motion_features.geolife import GeoLifeFeaturesExtraction


class MotionFeaturesExtraction:
    """ It implements the functions to extract motion features.
    
    Parameters
    ----------            
        No value
    """
    
    def _save_features(self, df_features, path_to_save, segment):
        """ This function saves the features extracted from trajectory.
        
        Parameters
        ----------
            df_features : pandas dataframe
                the segment to save
                
            path_to_save : str
                absolute path where to save the segments
                
            segment : str
                the user's name
                
        Returns
        -------
            no value
        """
        
        ## save data
        user_path, file_name = split(segment)
        user_name            = basename(dirname(user_path))
        transportation_name  = basename(user_path)
        
        ## where to save the features
        path_to_save = join(path_to_save, user_name, transportation_name, file_name)
        
        create_folder(path_to_save)
    
        df_features.to_csv(path_to_save, sep = ",", header = True, index = None)
        
        
        
    def get_features(self, transportation, folder_segments, folder_features):
        """ Receives the data about the segments, 
        organizes them in a list and extract the features from them, 
        saving it to the chosen folder.
        
        Parameters
        ----------
            transportation : list of str
                transportation modes which we want to extract features
                
            folder_segments : str
                absolute path where the segments to extract features are
                
            folder_features : str
                the folder where to save the features
                                
        Returns
        -------
            no value
        """
        
        segment_files = []
        
        ## Get the path to the transportation mode files (all of them)
        for transport in transportation:
            query               = transport + "*.csv"
            user_transportation = get_files(folder_segments, query, True)
            segment_files       = list(chain(segment_files, user_transportation))
        
        print("Processing {} segments...".format(len(segment_files)))
        
        for segment in segment_files:
            
            df_features = GeoLifeFeaturesExtraction.get_features(segment)
                                
            ## save data
            self._save_features(df_features, folder_features, segment)
