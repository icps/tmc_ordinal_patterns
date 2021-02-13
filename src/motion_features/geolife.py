# -*- coding: utf-8 -*-
import pandas as pd
import geopy.distance



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
            
            test_coords = self._test_coords(coords1[:2], coords2[:2])
            
            # if it is an acceptable latitude/longitude value
            if test_coords == True:
                distance = geopy.distance.distance(coords1, coords2).km
                
                distance_meters = distance * 1000 # to meters
            
                df_distance = df_distance.append({"distance": distance_meters}, 
                                                 ignore_index = True)
                
            else:
                print("invalid coordinates: {}".format(coords1, coords2))
        
        return df_distance        



class GeoLifeFeaturesExtraction:   
    
    def get_features(segment):
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
        
        spatial   = GeoLifeFeatures()
        
        extracted_features.append(latitude)
        extracted_features.append(longitude)
                
        ## extracted features
        df_distance = spatial.get_distance(latitude, longitude)        
        extracted_features.append(df_distance)
        
        
        for feat in extracted_features:
            feat.reset_index(drop = True, inplace = True)

        df_features = pd.concat(extracted_features, axis = 1)
        
        return df_features