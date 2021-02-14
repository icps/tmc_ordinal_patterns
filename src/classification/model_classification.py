# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import product

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import balanced_accuracy_score

np.random.seed(321)


class Classification:
    """ It implements classification functions
    
    Parameters
    ----------
        no value
    """
    
    
    def _prediction(self, X_train, X_test, y_train, y_test, model):
        """ It feds the array to the classifier
        
        Parameters
        ----------
            X_train : pandas dataframe
                X values for training dataset
            
            X_test : pandas dataframe
                X values for test dataset
                
            y_train : pandas dataframe
                y values for training dataset
                
            y_test : pandas dataframe
                y values for test dataset
                
            model : classifier
                the classifier used
            
        Returns
        -------
            y_pred : list of numeric
                predictions made by the classifier
        """
        
        model.fit(X_train, y_train.ravel())
        y_pred = model.predict(X_test)
        
        return y_pred   
    
    
    def _get_metrics(self, y_test, y_pred, metrics):
        """ It extract the classification metrics and confusion matrices 
        from the prediction.
        
        
        Parameters
        ----------
            y_train : pandas dataframe
                y values for training dataset
                
            y_test : pandas dataframe
                y values for test dataset
                
            metrics : str
                which set of metrics to apply
                can be: general, balanced, or perclass
                
        Returns
        -------
            accuracy : float
                accuracy value
            
            f1 : float
                F1-score value
            
            recall : float
                Recall value
            
            precision : float
                Precision value
            
            cm : ndarray of shape (n_classes, n_classes)
                Confusion matrix whose i-th row and j-th column entry indicates 
                the number of samples with true label being i-th class and 
                prediced label being j-th class (from sklearn page)
                (TL;DR: rows --> prediction, columns --> ground truth)
        """
        
        cm = confusion_matrix(y_test, y_pred)
        
        ## General Metrics
        if metrics == "general":
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average = "macro")
            recall = recall_score(y_test, y_pred, average = "macro")
            precision = precision_score(y_test, y_pred, average = "macro")
            
            ## micro
            # Calculate metrics globally by counting the total true positives, 
            # false negatives and false positives.

            ## macro
            # Calculate metrics for each label, and find their unweighted mean. 
            # This does not take label imbalance into account.
            
            return accuracy, f1, recall, precision, cm         
        
        ## Balanced Metrics
        if metrics == "balanced":
            accuracy = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average = "weighted")
            recall = recall_score(y_test, y_pred, average = "weighted")
            precision = precision_score(y_test, y_pred, average = "weighted")
            
            ## weighted
            # Calculate metrics for each label, and find their average weighted 
            # by support (the number of true instances for each label). 
            # This alters ‘macro’ to account for label imbalance; 
            # it can result in an F-score that is not between precision and recall.
            
            return accuracy, f1, recall, precision, cm 
        
        ## Per class Metrics
        if metrics == "perclass":
            precision, recall, f1, _ = precision_recall_fscore_support(y_test,
                                                                       y_pred, 
                                                        average = None)
            accuracy = accuracy_score(y_test, y_pred)
            
            ## Compute precision, recall, F-measure and support for each class
            # The support is the number of occurrences of each class in y_true
            # If average = None, the scores for each class are returned. 
            
            return accuracy, f1, recall, precision, cm 


    def _standardize_data(self, X):
        """ It standardizes data. 
        In case of multivariate data, this is done feature-wise 
        (in other words independently for each column of the data).
        
        Parameters
        ----------
            X : pandas dataframe
                dataframe of features 
                
        Returns
        -------
            X_stan : pandas dataframe
                datatrame of standardized features
        """
        
        X_stan = MinMaxScaler().fit_transform(X)
        
        string = "- Standardized data -"
        print(string)
        
        return X_stan

    

    def classification(self, X, y, model, standardize = True, n_folds = 10):
        """ It calculates metrics for classification prediction.
        
        Parameters
        ----------
            X : pandas dataframe
                dataframe of features
                
            y : pandas dataframe
                class labels
                
            model : classifier
                classifier model used to classify
                
            standardize : boolean (default = True)
                whether or not to standardize the features
                
            n_folds : int (default = 10)
                how many folders to divide data to cross-validation
        
        Returns
        -------
            df : pandas dataframe
                dataframe containing the metrics values for 
                general, balanced classes, and per class classification
                
            df_cm : list of array
                list containing the confusion matrices
        """
                
        col1 = ["general", "balanced", "perclass"]
        
        cvs = list(product(["CV"], range(1, n_folds + 1))) 
        col2 = [''.join([str(i) for i in p]) for p in cvs] + ["mean"]
        
        col3 = ["Accuracy", "F1", "Recall", "Precision"]
        
        df_results = {k: None for k in list(product(col1, col2))}
        
        df = pd.DataFrame(df_results, index = col3)
        
        df_cm = []

        kf = StratifiedKFold(n_splits = n_folds, shuffle = True)
        
        # Standardize
        if standardize == True:
            X = self._standardize_data(X)
        
        X_np = np.array(X)
        y_np = np.array(y)
                        
        for enum, (train, test) in enumerate(kf.split(X, y)):
                    
            X_train, X_test = X_np[train], X_np[test] 
            y_train, y_test = y_np[train], y_np[test]

            y_pred = self._prediction(X_train, X_test, y_train, y_test, model)
            
            acc, f1, rec, pre, cm = self._get_metrics(y_test, y_pred, "general")
            df.loc[:, ('general', 'CV' + str(enum + 1))] = acc, f1, rec, pre
            df_cm.append(cm)
            
            acc, f1, rec, pre, cm = self._get_metrics(y_test, y_pred, "balanced")
            df.loc[:, ('balanced', 'CV' + str(enum + 1))] = acc, f1, rec, pre
            
            acc, f1, rec, pre, cm = self._get_metrics(y_test, y_pred, "perclass")
            df.loc[:, ('perclass', 'CV' + str(enum + 1))] = acc, f1, rec, pre
                        
            
        for metric in col3:
            df = df.astype(object)
                        
            general_mean = sum(df.loc[metric, 'general'][:-1]) / n_folds
            df.loc[metric, ('general', 'mean')] = general_mean
                        
            balanced_mean = sum(df.loc[metric, 'balanced'][:-1]) / n_folds
            df.loc[metric, ('balanced', 'mean')] = balanced_mean
                        
            perclass_mean = np.sum(df.loc[metric, 'perclass'][:-1]) / n_folds
            df.loc[metric, ('perclass', 'mean')] = perclass_mean.tolist()
        
        # Confusion Matrix of all CVS: sum each col
        df_cm.append(np.sum(df_cm, axis = 0))
        
        # printing                
        print("GENERAL METRICS:")    
        print(df.loc[:, ('general', 'mean')])        
        
        return df, df_cm
        
        
