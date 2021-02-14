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
    
    wdir                  = join("db", "GeoLife", "example", "")
       
    ## folders
    folder_dataset        = join(wdir, "Data", "")
    folder_segments       = join(wdir, "segments", "")
    folder_features       = join(wdir, "motion_features", "")
    folder_op             = join(wdir, "op_features", "")
    folder_classification = join(wdir, "classification")
    
    ## parameters    
    transportation        = ['walk', 'bus', 'car', 'bike', 'taxi']
    
    ## [D, tau]
    op_values             = [[3, 1]]
    
    ## features
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

        model = KNeighborsClassifier(n_neighbors = 2)

        print("CLASSIFIER: {}".format(type(model).__name__))

        op = OPClassification(op_values, motion_features, folder_op, folder_features, 
                                folder_classification, model, op_features) 
        op.classification(n_folds = 2)
    
        
        
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