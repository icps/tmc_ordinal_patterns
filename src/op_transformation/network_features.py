# -*- coding: utf-8 -*-


import numpy as np
import networkx as nx


class NetworkFeatures:
    """ Extracts Network Features from 
    OP histogram of probability distribution.
    
    Parameters
    ----------
        no value
    """
        
    def probability_self_transition(self, G):
        """ Extracts the probability of self-transition.
        
        Parameters
        ----------
            G : networkx graph
                OP transition network (graph)
            
        Returns
        -------
            prob_self_loop : float
                the probability of self-transition
        """
        
        nodes_self_loops = list(nx.selfloop_edges(G, 'weight'))
        self_weights     = [w[2] for w in nodes_self_loops]
        prob_self_loop   = np.sum(self_weights)
        
        return prob_self_loop
    
    

