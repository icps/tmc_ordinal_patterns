# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import networkx as nx

from collections import Counter
from itertools import permutations, groupby

from src.op_transformation.IT_features import InformationTheoryFeatures
from src.op_transformation.network_features import NetworkFeatures

np.random.seed(321)



class OPTransformation:
    
    def get_transformation(self, data, D, tau):
        """ Receives the trajectory and transform it to OP transformation.
        
        Parameters
        ----------
            data : pandas dataframe
                the time series to calculate OP
                
            D : int
                the embedding dimension D
                
            tau : int
                the embedding delay tau
            
        Returns
        -------
            symbols : list of str
                a list containing the OP symbols extracted from the time series
        """
    
        symbols = []
    
        # traversing the whole time series
        for s in range(0, data.size - (D - 1)*tau): # complexity: O(data.size)
    
            # The correct is s + (D - 1)*tau, but python counts from zero 
            # and range is exclusive, so 
            # if s = 0 --> ind = range(s, 1 + s + (D - 1)*tau, tau) 
            # => ind = range(0, 3, step = 1) => ind = (0, 1, 2)
            ind       = range(s, 1 + s + (D - 1)*tau, tau)
    
            sub       = data.iloc[ind]
            op_window = np.argsort(sub)
            
            # op_window returns an array([2, 1, 0])
            # we add 1 cuz our pattern starts from 1 
            # (it is just style, it can be from 0, but we prefer from 1 to D)
            # and we join to get '321' from ['3', '2', '1']
            op_pattern = ''.join(map(str, op_window + 1))
    
            symbols.append(op_pattern)
    
        return symbols
    
    
    def get_permutations(self, D):
        """ Calculates the permutation of D values. Ex:
            D = 2 --> ['12' '21']
            
        Parameters
        ----------
            D : int
                the embedding dimension D
            
        Returns
        -------
            op_permutation : list of str
                a list containing the symbols of D! permutation
        """
        
        # range in python is exclusive, so we add 1 to include D value
        # permutations return a tuple
        # we convert from (3, 2, 1) to '321' similar to the above function
        op_permutation = [''.join(map(str, p)) for p in permutations(range(1, D + 1))]
        op_permutation = sorted(op_permutation)
    
        return op_permutation
    
    
    
    def get_probabilities(self, frequency):
        """ Tranform the frequency distribution to probability distribution
        
        Parameters
        ----------
            frequency : list of str
                a list containing the OP frequency distribution 
            
        Returns
        -------
            probs : list of float
                a list containing the OP probability distribution
        """
        
        probs = [x/sum(frequency) for x in frequency]
        
        return probs
        
    
    
    def get_distribution(self, symbols, symbol_permutation, D, tau, numred = False, prob = False):
        """ Receives the trajectory and transform it to OP frequency distribution.
        
        Also, this function can calculate a Numerosity Reduction to OP transformation:
            if there is repeated symbols in a row, it is replaced as one appearance. 
            Ex: (321, 321, 321, 213, 321) --> (321, 213, 321)
        
        Parameters
        ----------
            symbols : list of str
                a list containing the OP symbols extracted from the time series
                
            symbol_permutation : list of str
                a list containing the symbols of D! permutation
                
            D : int
                the embedding dimension D
                
            tau : int
                the embedding delay tau
                            
            numred : bool
                if True, Numerosity Reduction is applied
                
            prob : bool
                if True, returns OP histogram of probability distribution
                if False, returns OP histogram of frequency distribution
            
        Returns
        -------
            sorted_freq : list of int/float
                a list containing the OP frequency or probability distribution 
        """
    
        if (numred == False):
            dpi = Counter(symbols)
    
        else:
            num_red = [k for k, g in groupby(symbols)]
            dpi = Counter(num_red)
    
        # Get the patterns and their frequencies in OP transformation
        patterns    = list(dpi.keys())
        frequencies = list(dpi.values())
    
        # Some pattern are missing. 
        # So, we add this missing patterns with zero frequency.    
        complete_patterns = []
        missing_patterns = list(set(symbol_permutation) - set(patterns))
    
        complete_patterns.append(patterns + missing_patterns)
        complete_patterns.append(frequencies + [0] * len(missing_patterns))
    
        # Sort values based on the patterns sorted (so every list is in the same order)
        sorted_freq = [x for _,x in sorted(zip(complete_patterns[0], complete_patterns[1]))]
        
        if prob == True:
            sorted_freq = self.get_probabilities(sorted_freq)
    
        return sorted_freq
       
    
    
    def get_transition_matrix(self, symbols, symbol_permutation, D, tau,
                              normalized = True, loop = True):
        """ Receives the OP symbols from trajectory and transform it to OP transition matrix.
        
        Parameters
        ----------
            symbols : list of str
                a list containing the OP symbols extracted from the time series
                
            symbol_permutation : list of str
                a list containing the symbols of D! permutation
                
            D : int
                the embedding dimension D
                
            tau : int
                the embedding delay tau
                            
            normalized : bool
                if True, the matrix is normalized
                
            loop : bool
                if True, allows self-transition
            
        Returns
        -------
            transition_matrix : list of lists (matrix)
                the transition matrix
        """
        

        # filling the transition matrix with zeros
        n                 = len(symbol_permutation)
        transition_matrix = pd.DataFrame(np.zeros((n, n)), 
                                         index   = symbol_permutation, 
                                         columns = symbol_permutation)
        
        #transition.loc[row, column]
        for val in range(0, len(symbols) - 1):
            row = symbols[val]
            col = symbols[val + 1]
            
            if (row != col or loop == True): # allow loops
                previous_value = transition_matrix.loc[[row], [col]]
                transition_matrix.loc[[row], [col]] = previous_value + 1
                
        if normalized == True:
            transition_matrix = transition_matrix/transition_matrix.sum()
            
        return transition_matrix
    
    
    
    def get_transition_network(self, transition_matrix):
        """ Receives the OP transition matrix to Ordinal Patterns Transition Network (OPTN).
        
        Parameters
        ----------
            transition_matrix : list of lists (matrix)
                the transition matrix from OP transformation
            
        Returns
        -------
            G : networkx graph
                the OPTN
        """
         
        G = nx.from_numpy_matrix(transition_matrix.values, create_using=nx.DiGraph)
        
        # node labels
        name            = list(transition_matrix.columns)
        label_mapping   = dict(list(enumerate(name)))
        G               = nx.relabel_nodes(G, label_mapping)
        
        # remove nan edges
        edge_weights    = nx.get_edge_attributes(G, 'weight')
        edges_to_remove = (e for e, w in edge_weights.items() if np.isnan(w)) 
        G.remove_edges_from(edges_to_remove)
        
        return G

    

    def get_features(self, op_features, data, D, tau, numred, prob):
        """ Calculates the Information Theory and Network Features from OP Transformation
        
        Parameters
        ----------
            op_features : list of str
                which features to extract from OP and OPTN transformation
            
            data : pandas dataframe
                the time series to calculate OP
                
            D : int
                the embedding dimension D
                
            tau : int
                the embedding delay tau
                
            numred : bool
                if True, Numerosity Reduction is applied in distribution
                
            prob : bool
                if True, returns OP histogram of probability distribution
                if False, returns OP histogram of frequency distribution
                
        Returns
        -------
            pe, sc, st : tuple of float
                Permutation Entropy, Statistical Complexity, 
                and Probability of Self-transition values
        """
        
        feature_list = []
        
        n = len(data)
        if (D - 1)*tau < n:
                       
            ## Transform data to op histograms
            symbols            = self.get_transformation(data, D, tau)
            symbol_permutation = self.get_permutations(D)
            histogram          = self.get_distribution(symbols, symbol_permutation, D, tau, numred, prob)
            

            #### OP Features
            itf                = InformationTheoryFeatures(prob)
            
            # Permutation Entropy
            pe                 = itf.permutation_entropy(histogram, normalized = True)
            feature_list.append(pe)
                
            # Statistical Complexity
            sc                 = itf.statistical_complexity(histogram, pe)
            feature_list.append(sc) 
            
            
            ##### OPTN Features
            netf               = NetworkFeatures()
        
            transition_matrix  = self.get_transition_matrix(symbols, symbol_permutation, D, tau, normalized = True)
            G                  = self.get_transition_network(transition_matrix)
            
            # Probability of self-transition
            st                 = netf.probability_self_transition(G)
            feature_list.append(st) 
                    
            
        else:
            print("This D and tau value needs trajectories bigger than {}".format(n))
            pass
        
        return feature_list
        
        
        
        
        
        
            
            
            
            
        
        
        
        
        
        
        
        
    
    

