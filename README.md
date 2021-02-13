# Leveraging the Self-Transition Probability of Ordinal Patterns Transition Network for Transportation Mode Identification Based on GPS Data
------

## Abstract

The analysis of GPS trajectories is a well-studied problem in Urban Computing and has been used to track people. 
Analyzing people mobility and identifying the transportation mode used by them is essential for cities that want to reduce traffic jams and travel time between their points, thus helping to improve the quality of life of citizens. The trajectory data of a moving object is represented by a discrete collection of points through time, i.e., a time series. 
Regarding its interdisciplinary and broad scope of real-world applications, it is evident the need of extracting knowledge from time series data. 
Mining this type of data, however, faces several complexities due to its unique properties. 
Different representations of data may overcome this. 
In this work, we propose the use of a feature retained from the Ordinal Pattern Transition Graph, called the probability of self-transition for transportation mode classification. 
The proposed feature presents better accuracy results than Permutation Entropy and Statistical Complexity, even when these two are combined. 
This is the first work, to the best of our knowledge, that uses Information Theory quantifiers to transportation mode classification, showing that it is a feasible approach to this kind of problem.

The paper is available at: _link_

If you find this code useful, please consider citing our paper: _bib citation_

## Framework Overview

Our framework is composed of the following steps: Data Segmentation, Feature Extraction, and Classification, as shown in the following figure. Each step is better explained in our paper.

![Framework Overview](https://github.com/icps/tmc_ordinal_patterns/blob/main/FrameworkTMC.png)


## About the Code

### Highlights

- We present an easy Python implementation of Ordinal Patterns (OP) transformation, as well as the OP transition network (OPTN).

- The preprocessing code is easy to migrate to another dataset. Also, it is easy to add new features to extract.

- All the experiments presented in the paper can be reproducible using the provided code.

### Software Requirements

This code is tested on Linux Ubuntu 18.04.03 and Debian 10. We use Python (version 3.7.3) with the Anaconda distribution (version 4.7.11). Prior to running the experiments, make sure to install the following required libraries: 

- [scikit-learn](https://scikit-learn.org/stable/) (version 0.23.2)
- [GeoPy](https://geopy.readthedocs.io/en/stable/) (version 1.20.0)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/) (version 0.90)
- [Pandas](https://pandas.pydata.org/) (version 0.24.2)
- [Numpy](https://numpy.org/) (version 1.16.4)

Note that our code is not developed focused on speed. Several enhancements can be made on this matter.


Additionally, it is necessary to download the GeoLife dataset in this [link](https://www.microsoft.com/en-us/download/details.aspx?id=52367). Move the folder called `Data/` into the project folder `./db/GeoLife/`.

### Project File Structure

- Please create a folder called `db/GeoLife/` under our main folder and put the `Data/` folder from GeoLife dataset;
  - As you ran each step of our framework, new folders to save the files will be created in `db/` folder, such as `segments/` (to save the segmented trajectories), `motion_features/` (to save the features extracted from the segments), `op_features/` (to save the OP transformation), and `classification_<current_date>/` (to save the classification results; the current date is added so the classification results are not override.
  
- The `src/` folder contains the following folders:
  - `utilities/` contains some useful python codes to handle folders and subfolders (create and get their files);
  - `motion_features/` contains the code to extract the motion features from the segmented data. In this work we only extract distance, but it is easy to extend the code to another features;
  - `segmentation/` contains the preprocessing code to manipulate the GeoLife dataset. If you want to perform the experiments into another dataset, you can base your code in the `geolife.py` file;
  - `op_transformation/` contains the codes to transform data in OP and OPTN transformations, as well as the codes to extract the features explored in this work (permutation entropy, statistical complexity, and probability of self-transition);
  - `classification/` contains the code to classify using the sklearn-based model provided.
  
- The `main.py` file run all steps needed to the experiments. It may take some days to run. Also, you can run each step separately, given that you already have the needed data (e.g., you can apply the OP transformation if you already have the motion_features). To change which step will run, open the file and change the code in the main function (lines 122 to 125) to _True_ in the step you want to run.

## Usage example

As the full experiment can take a few days to run completely, we recommend you to first execute the demo version, that follows a procedure very similar to the real experiment. To this, you can run the following code:

**TODO: add example**

## Support

Feel free to send any questions, comments, or suggestions (in english or portuguese) to Isadora Cardoso-Pereira (isadoracardoso@dcc.ufmg.br).
