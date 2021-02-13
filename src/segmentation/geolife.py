# -*- coding: utf-8 -*-

""" 
Created on Mon March 23 14:17:00 2020

@author: isadora cardoso (/icps)
"""

import numpy as np
import pandas as pd
from os.path import dirname, basename, join
from src.utilities.utils import get_files, create_folder


class GeoLifeSegmentation:
    """ These functions are implemented specifically to read from GeoLife dataset.

    GeoLife dataset has the following format:
        
    Data/
    └── <user-number>/
        ├── labels.txt
        └── Trajectory/
            └── <trajectories>.plt
    
    where <user-number> and <trajectories> names vary.
    
    Here, for each <user-number>, we segment each trajectory, based on the labels.
    """


    def _read_label(self, path):
        """ Reads the label.txt of a user.
        It contains the start point, the end point, and the transportation mode 
        used in each trajectory of a user.
        
        Parameters
        ----------
            path : str
                the path where the label.txt is.
            
        Returns
        -------
            df_labels : pandas dataframe    
                dataframe containing the start point, end point, 
                and transportation mode information.
        """
        
        ## read data
        df_labels = pd.read_csv(path, sep = "\t", header = 0)
        df_labels.columns = ['start', 'end', 'transportation']
    
        ## convert start and end to date type
        df_labels['start'] = pd.to_datetime(df_labels['start'])
        df_labels['end'] = pd.to_datetime(df_labels['end'])
        
        return df_labels



    def _read_trajectory(self, path):
        """ Reads a trajectory 
        (specifically latitude, longitude, altitude, date, and time)
        and returns an organized dataframe.

        
        Parameters
        ----------
            path : str
                the path where the trajectory is.
        
        Returns
        -------
            df_trajectory : pandas dataframe
                dataframe containing latitude, longitude, altitude, and timestamp.
                timestamp is date and time as following: <date time>.
        """
         
        ## read data
        # the first six lines are information that does not matter here
        # columns 0, 1 and 3 are latitude, longitude, and altitude, respectively
        # column 5 and 6 are date and time, respectively
        data = np.genfromtxt(path, delimiter = ',', skip_header = 6, 
                                      usecols = (0, 1, 3, 5, 6), dtype = None)
    
        df_trajectory = pd.DataFrame(data)
        df_trajectory.columns = ['latitude', 'longitude', 'altitude', 'date', 'time']
    
        ## convert date and time from bytes to str
        df_trajectory["date"] = [str(x, "utf-8") for x in df_trajectory["date"]]
        df_trajectory["time"] = [str(x, "utf-8") for x in df_trajectory["time"]]
    
        # join date and time into single column <date time>
        df_trajectory['timestamp'] = df_trajectory[['date', 'time']].apply(lambda x: ' '.join(x), 
                                                         axis = 1)
    
        # drop separated date and time
        df_trajectory = df_trajectory.drop(['date', 'time'], axis = 1)
    
        ## convert timestamp to date type
        df_trajectory['timestamp'] = pd.to_datetime(df_trajectory['timestamp'])
        
        return df_trajectory
      
            
    
    def _get_segments(self, df_trajectory, df_label, min_points):
        """ Segments the trajectory and 
        associates the user's GPS information with the transportation mode.
        
        Parameters
        ----------
            df_user : pandas dataframe
                the GPS information of the user.
                
            df_label : pandas dataframe
                the transportation mode information.
                
            min_points : int
                segments with less than <min_point> will be ignored
                
        Returns
        -------
            segments : dict of pandas dataframe
                dictionary containing the segments of each trajectory
        """
        
        segments = []
        
        for i in range(0, len(df_label)):
            
            start_time = (df_trajectory['timestamp'] >= df_label['start'][i])
            end_time   = (df_trajectory['timestamp'] <= df_label['end'][i])
            
            trajectory = df_trajectory.loc[start_time & end_time]
            trajectory = trajectory.reset_index(drop = True)
            
            if len(trajectory) > min_points:                
                label = df_label['transportation'][i]
                segments.append((label, trajectory))
            
        return segments
    
    
    
    def _save_segments(self, transport_name, segment, folder_segments, user_name):
        """ Saves the segments extracted from trajectory.
        
        Parameters
        ----------
            transport_name : str
                the transportation used in the trajectory
        
            segment : pandas dataframe
                the segment to save
                
            path_to_save : str
                absolute path where to save the segments
                
            user_name : str
                the user's name
                
        Returns
        -------
            no value
        """
        
        path_to_save = join(folder_segments, user_name)
        
        ## write to file
        previous_traj = get_files(path_to_save, transport_name, False)
        k = len(previous_traj) + 1
        
        ## where to save
        path_to_save = join(path_to_save, transport_name, "")
        create_folder(path_to_save)
        
        ## name to save
        file_name = "{}_{:03d}.csv".format(transport_name, k)
        save_in = join(path_to_save, file_name)
        
        segment.to_csv(save_in, sep = ",", header = True, index = None)
                
        
        
    
    def segmentation(self, folder_dataset, folder_segments, min_points = 10):
        """ Receives the raw data from GeoLife dataset and 
        segmentates it, saving it to the chosen folder.
        
        Parameters
        ----------
            folder_dataset : str
                absolute path where the dataset is
            
            folder_segments : str
                absolute path where to save the segments
                
            min_points : int
                segments with less than <min_point> will be ignored
            
        Returns
        -------
            no value
        """
        
        ## get user's path who have transportation mode information
        # (i.e., only the users with the "labels.txt")
        label_users = get_files(folder_dataset, "labels.txt", True)
        
        n           = len(label_users)
                
        for enum, current_label in enumerate(label_users):
            
            print("{:02d} of {} users -- processing user {}".format(enum + 1, n, dirname(current_label)))
            
            # read label information
            df_label    = self._read_label(current_label)
            
            ## read user's trajectories data
            user_folder = dirname(current_label)
            user_name   = basename(user_folder)
            user_files  = get_files(user_folder, ".plt", True)

            for current_trajectory in user_files:
                df_user      = self._read_trajectory(current_trajectory)
                trajectories = self._get_segments(df_user, df_label, min_points)
                
                for transport, trajectory in trajectories:
                    self._save_segments(transport, trajectory, folder_segments, user_name)
