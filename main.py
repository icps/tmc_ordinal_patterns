#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 17:51:39 2020

@author: isadora
"""

## imports
import random
from os.path import join
from datetime import datetime

## Segmentation
from src.segmentation.geolife import GeoLifeSegmentation

## Extracting of Motion Features
from src.motion_features.extraction import MotionFeaturesExtraction

## OP and OPTN Transformation and Extracting of Information Theory Features
from src.op_transformation.extraction import OPTransformationExtraction

### Classification
from src.classification.op import OPClassification

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

seed = 321
random.seed(seed)



def framework_geolife(segmentation, motion, op, classification):
    
    print("--- EXECUTING THE FRAMEWORK IN GEOLIFE DATASET ---")
    
    start_time = datetime.now()
    print("started at {}".format(start_time))
    
    wdir = join("db", "GeoLife", "")
       
    ## folders
    folder_dataset        = join(wdir, "Data", "")
    folder_segments       = join(wdir, "segments", "")
    folder_features       = join(wdir, "motion_features", "")
    folder_op             = join(wdir, "op_features", "")
    folder_classification = join(wdir, "classification")
    
    ## parameters    
    transportation        = ['walk', 'bus', 'car', 'bike', 'taxi']
    
    ## [D, tau]
    op_values             = [[3, 1], [4, 1], [5, 1], [6, 1]]

    
    motion_features       = ["latitude", "longitude", "distance"]
    op_features           = ["permutation_entropy", "statistical_complexity", "self_probability"]
    
    ## Segmentation
    if segmentation == True:
        print("----- SEGMENTING DATA")
        GeoLifeSegmentation().segmentation(folder_dataset, folder_segments)
    
    ## Motion Features Extraction
    if motion == True:
        print("----- CALCULATING MOTION FEATURES")
        mfeatures = MotionFeaturesExtraction()
        mfeatures.get_features(transportation, folder_segments, folder_features)
    
    ## OP and OPTN Transformation
    if op == True:
        print("----- CALCULATING INFORMATION THEORY FEATURES")
        op = OPTransformationExtraction()
        op.get_transformation(op_values, motion_features, op_features,
                              transportation, folder_features, folder_op)

    ## Classification
    if classification == True:
        print("----- CLASSIFYING")

        knn         = KNeighborsClassifier(n_neighbors = 2)
        
        svm_r       = SVC(kernel = "rbf")
        svm_l       = SVC(kernel = "linear")
        
        dt          = DecisionTreeClassifier(random_state = 321)
        
        trees       = 50
        rf          = RandomForestClassifier(n_estimators = trees, random_state = 321, n_jobs = -1)
        xgboost     = XGBClassifier(n_estimators = trees, random_state = 321, n_jobs = -1)
                
        models      = [knn, svm_r, svm_l, dt, rf, xgboost]
#         models = [rf]
        
        op1         = ["permutation_entropy"]
        op2         = ["statistical_complexity"]
        op3         = ["self_probability"]
        op_features = [op1, op2, op3, op1 + op2, op1 + op2 + op3]
#         op_features = [op_features]
        
        for opfeat in op_features:
        
            for model in models:

                print("CLASSIFIER: {}".format(type(model).__name__))

                op = OPClassification(op_values, motion_features, folder_op, folder_features, 
                                        folder_classification, model, opfeat)
                
                n_folds = 10
                op.classification(n_folds, transportation)
    
        
        
    end_time   = datetime.now()
    total_time = end_time - start_time
    print("finished at {}".format(end_time))
    print("Total time: {}".format(total_time))


#######################################################################
#######################################################################


def main():

    segmentation   = True
    motion         = True
    op             = True
    classification = True
    
    framework_geolife(segmentation, motion, op, classification)
    

if __name__ == '__main__':
    main()
