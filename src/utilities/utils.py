#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:11:09 2019

@author: isadora
"""

from glob import glob
import os


def create_folder(filename):
    """ This function creates folders and subfolders
    
    Parameters
    ----------
        filename : str
            the path to create the folders
            
    Returns
    -------
        no value 
    """
    
    folder = os.path.dirname(filename)

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    

def get_files(path, search_pattern, rec = True):
    """ This function lists all the files in a folder.
    
    Parameters
    ----------
        path : str
            the path which is the folder.
            
        search_pattern : str
            the files must contain this string.
            
        rec : bool
            if it searches recursively into the folder or not.
    
    Returns
    -------
        files : list of str
            absolute path of all files in a folder.
    """

    query = os.path.join(path, "**", "*" + search_pattern + "*")
    files = [f for f in glob(query, recursive = rec)]
    
    return sorted(files)