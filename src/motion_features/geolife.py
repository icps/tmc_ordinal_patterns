# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import geopy.distance
from datetime import datetime



class GeoLifeFeatures:
    """ Implements the functions to extract motion features from GPS data.
    Here, such data is provided by GeoLife dataset.
    
    Parameters
    ----------
        no value    
    """
   
    
    # latitude must be in [-90, 90]
    # longitude must be in [-180, 180]
    def _test_coords(self, coords1, coords2):
        """ Tests if latitude and longitude are in an acceptable range
        latitude must be in [-90, 90]
        and longitude must be in [-180, 180]
        
        Parameters
        ----------
            coords1 : tuple of float
                latitude and longitude values of coordinate 1
                
            coords2 : tuple of float
                latitude and longitude values of coordinate 2
        
        Returns
        -------
            bool
                if the values are in the acceptable range, returns True
                if it is not, returns False
        """
    
        lat1, long1 = float(coords1[0]), float(coords1[1])
        lat2, long2 = float(coords2[0]), float(coords2[1])
        
        tests = {}
        
        tests["lat1"]  = (-90 <= lat1 <= 90)
        tests["long1"] = (-180 <= long1 <= 180)
        
        tests["lat2"]  = (-90 <= lat2 <= 90)
        tests["long2"] = (-180 <= long2 <= 180)
        
        for values in tests.values():
            if values == False:
                return False
            
        return True
    
    
    # seconds
    def _get_diff_time(self, timestamp):
        """ Calculates the time difference between every two points.
        
        Parameters
        ----------
            timestamp : pandas dataframe
                a dataframe from trajectory containing the time information
            
        Returns
        -------
            time : pandas datframe
                a dataframe containing the time difference between points
        """
        
        time = []
        
        for i in range(0, (len(timestamp) - 1)):
            
            time1 = datetime.strptime(timestamp[i], "%Y-%m-%d %H:%M:%S")
            time2 = datetime.strptime(timestamp[i+1], "%Y-%m-%d %H:%M:%S")
                        
            diff_time = (time2 - time1)
            
            diff_time_seconds = diff_time.total_seconds()
    
            # if want to convert seconds to hours
    #            diff_time_hours = diff_time.total_seconds() / 3600
            
            time.append(diff_time_seconds)
            
        return time
    
    
    # m/s    
    def get_speed(self, distance, timestamp):
        """ Calculates speed
        
        Parameters
        ----------
            distance : pandas dataframe
                dataframe containing distance in meters
                
            timestamp : pandas dataframe
                dataframe containing time in seconds
                
        Returns
        -------
            df_speed : pandas dataframe
                dataframe containing the speed between points, 
                in meters per second        
        """
        
        df_speed = pd.DataFrame(columns = ["speed"])
        
        diff_time = self._get_diff_time(timestamp)
    
        for dist, time in zip(np.array(distance), diff_time):
            
            if time == 0:
                speed = 0
            
            else:
                speed = float(dist)/float(time)
        
            df_speed = df_speed.append({"speed": speed}, ignore_index = True)
              
        return df_speed
    
    
    
    # m/s^2
    def get_acceleration(self, speed, timestamp):
        """ Calculates acceleration
        
        Parameters
        ----------
            speed : pandas dataframe
                dataframe containing speed in meters per second
                
            timestamp : pandas dataframe
                dataframe containing time in seconds
                
        Returns
        -------
            df_acceleration : pandas dataframe
                dataframe containing the acceleration between points, 
                in meters per second squared    
        """
        
        df_acceleration = pd.DataFrame(columns = ["acceleration"])
        
        diff_time = self._get_diff_time(timestamp)
        
        aux = pd.DataFrame([0], columns = ['speed'])
        speed = aux.append(speed, ignore_index = True)
    
        for i in range(0, (len(speed) - 1)):
            
            if diff_time[i] == 0:
                acceleration = 0
                
            else:
                delta_speed = speed.loc[i+1] - speed.loc[i]
                
                # acceleration or desacceleration (can be negative)
                acceleration = delta_speed / diff_time[i]
                acceleration = acceleration.values[0]
        
            df_acceleration = df_acceleration.append({"acceleration": acceleration}, 
                                                     ignore_index = True)  
        
        return df_acceleration
    
    
   # https://gist.github.com/jeromer/2005586
    def _calculate_initial_compass_bearing(self, pointA, pointB):
        """ Calculates the bearing between two points.
        
        Parameters
        ----------
            pointA : tuple
                latitude/longitude for the first point, in decimal degrees
                
            pointB : tuple
                latitude/longitude for the second point, in decimal degrees
                                
        Returns
        -------
            compass_bearing : float
                bearing in degress
        """
        
        if (type(pointA) != tuple) or (type(pointB) != tuple):
            raise TypeError("Only tuples are supported as arguments")
    
        lat1 = math.radians(pointA[0])
        lat2 = math.radians(pointB[0])
    
        diffLong = math.radians(pointB[1] - pointA[1])
    
        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
                * math.cos(lat2) * math.cos(diffLong))
    
        initial_bearing = math.atan2(x, y)
    
        # Now we have the initial bearing but math.atan2 return values
        # from -180° to + 180° which is not what we want for a compass bearing
        # The solution is to normalize the initial bearing as shown below
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360
    
        return compass_bearing 
    
    
    # direction
    def get_bearing(self, latitude, longitude):
        """ Calculates bearing
        
        Parameters
        ----------
            latitude : pandas dataframe
                dataframe containing latitude in decimal degrees
                
            longitude : pandas dataframe
                dataframe containing longitude in decimal degrees
                
        Returns
        -------
            df_bearing : pandas dataframe
                dataframe containing the bearing between two set of points,
                in degrees        
        """
        
        df_bearing = pd.DataFrame(columns = ["bearing"])
        
        for i in range(0, (len(latitude) - 1)):
                          
            coords1 = (latitude[i], longitude[i])
            coords2 = (latitude[i + 1], longitude[i + 1])
                        
#             test_coords = self._test_coords(coords1, coords2)
            
            # if it is an acceptable latitude/longitude value, calculates it
            # if not, does not calculate using such points
#             if test_coords == True:
            try:
                bearing = self._calculate_initial_compass_bearing(coords1, coords2)
            
                df_bearing = df_bearing.append({"bearing": bearing}, ignore_index = True)
                
#             else:
            except:
                print("invalid coordinates: {}".format(coords1, coords2))
            
        return df_bearing
    
        
    
    # meters
    def get_distance(self, latitude, longitude):
        """ Calculates distance
        
        Parameters
        ----------
            latitude : pandas dataframe
                dataframe containing latitude in decimal degrees
                
            longitude : pandas dataframe
                dataframe containing longitude in decimal degrees
                
        Returns
        -------
            df_distance : pandas dataframe
                dataframe containing the distance between two set of points,
                in meters        
        """
        
        df_distance = pd.DataFrame(columns = ["distance"])
        
        for i in range(0, (len(latitude) - 1)):
                            
            coords1 = (latitude[i], longitude[i])
            coords2 = (latitude[i+1], longitude[i+1])
            
#             test_coords = self._test_coords(coords1[:2], coords2[:2])
            
            # if it is an acceptable latitude/longitude value
#             if test_coords == True:
            try:
                distance = geopy.distance.distance(coords1, coords2).km
                
                distance_meters = distance * 1000 # to meters
            
                df_distance = df_distance.append({"distance": distance_meters}, ignore_index = True)
                
#             else:
            except:
                print("invalid coordinates: {}".format(coords1, coords2))
        
        return df_distance        



class GeoLifeFeaturesExtraction:   
    
    def get_features(segment, motion_features):
        """ Extracts features from GeoLife data and returns an organized dataframe.
        
        Parameters
        ----------
            segments : pandas dataframe
                the segment to calculate features
                
        Returns
        -------
            df_features : pandas dataframe
                dataframe containing all features of a segment
        """
          
        df_segment = pd.read_csv(segment, sep = ",", header = 0)
        
        extracted_features = []
        
        # from dataset
        latitude  = df_segment['latitude']
        longitude = df_segment['longitude']
        timestamp = df_segment['timestamp']
        
        spatial   = GeoLifeFeatures()
        
        if "latitude" in motion_features:
            extracted_features.append(latitude)
            
        if "longitude" in motion_features:
            extracted_features.append(longitude)
                
        ## extracted features
#         df_distance = spatial.get_distance(latitude, longitude)        
#         extracted_features.append(df_distance)
        
        ## extracted features
        if "distance" in motion_features:
            df_distance = spatial.get_distance(latitude, longitude)
            extracted_features.append(df_distance)
        
        if "bearing" in motion_features:
            df_bearing = spatial.get_bearing(latitude, longitude)
            extracted_features.append(df_bearing)
            
        if "speed" in motion_features:
            if "distance" not in motion_features:
                df_distance = spatial.get_distance(latitude, longitude)           
            
            df_speed = spatial.get_speed(df_distance, timestamp)
            extracted_features.append(df_speed)
            
        if "acceleration" in motion_features:
            if "speed" not in motion_features:
                df_speed = spatial.get_speed(df_distance, timestamp)
                
            df_acceleration = spatial.get_acceleration(df_speed, timestamp)
            extracted_features.append(df_acceleration)
        
        
        for feat in extracted_features:
            feat.reset_index(drop = True, inplace = True)

        df_features = pd.concat(extracted_features, axis = 1)
        
        return df_features