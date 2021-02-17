#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 20:12:39 2020

@author: isadora
"""

import math

class InformationTheoryFeatures:
    """ Extracts Information Theory Features from 
    OP histogram of probability distribution.
    
    Parameters
    ----------
        prob : bool
            if True, it is OP histogram of probability distribution
            if False, it is OP histogram of frequency distribution
    """
    
    def __init__(self, prob):
        self.prob = prob
                
        
    def _get_probs(self, op):
        """ Tranform the frequency distribution to probability distribution
        
        Parameters
        ----------
            op : list of int/float
                OP histogram of probability or frequency distribution
            
        Returns
        -------
            probs : list of float
                OP histogram of probability distribution
        """
        if self.prob == False:
            probs = [x/sum(op) for x in op]
        else:
            probs = self.prob
        
        return probs
    
    

    def permutation_entropy(self, op, normalized = False):
        """ Calculates the Permutation Entropy (PE)
        
        Parameters
        ----------
            op : list of int/float
                OP histogram of probability or frequency distribution
        
            normalized : bool
                if True, calculates the normalized PE
        
        Returns
        -------
            pe : float
                PE value
        """
        
        probs = self._get_probs(op)
        
        pe  = 0
        for p in probs:
            if p > 0:
                pe -= p*math.log(p)
        
        if (normalized == True):
            pe = pe/math.log(len(probs))
            
        return pe
        
    
    def statistical_complexity(self, op, entropy):
        """ Calculates the Statistical Complexity (SC)
        
        Parameters
        ----------
            op : list of int/float
                OP histogram of probability or frequency distribution
        
            entropy : float
                Permutation Entropy value (normalized or not)
        
        Returns
        -------
            C : float
                SC value
        """
        
        if entropy is None:
            entropy = self.permutation_entropy(op)
        
        probs = self._get_probs(op)
        
        # the length of the probabilities, 
        n = len(op)
        
        # the reference distribution (uniform)
        P_u = [1/n]*n
        
        # the Jensen-shannon divergence
        pe_pu =  self.permutation_entropy(P_u)/2
        pe_op = self.permutation_entropy(probs)/2
        
        aux      = [(x + y)/2 for x, y in zip(probs, P_u)]
        pe_op_pu = self.permutation_entropy(aux)
        
        JS = pe_op_pu - pe_op - pe_pu
        
        # the statistical complexity
        # math.log is ln
        aux = (((n+1)/n) * math.log(n + 1) - 2*math.log(2*n) + math.log(n))
        Q_0 = -2*(1/aux)
        Q = Q_0 * JS
        C = Q*entropy
        
        return C
    


