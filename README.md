# Leveraging the Self-Transition Probability of Ordinal Patterns Transition Network for Transportation Mode Identification Based on GPS Data

## Abstract

Analysing people mobility and identifying the transportation mode used by them is essential for cities that want to reduce traffic jams and travel time between their points, thus helping to improve the quality of life of citizens. Mining this type of data, however, faces several complexities due to its unique properties. In this work, we propose the use of Information Theory quantifiers retained from the Ordinal Patterns (OP) transformation, for transportation mode identification. As an initial exploration, our results show that OP satisfactorily characterises the trajectories. Moreover, in this scenario, the characteristics of OP transformation can be advantageous, such as its simplicity, robustness, and speed. 


The paper is available at: https://arxiv.org/abs/2007.08687

If you find this code useful, please consider citing our paper: 

```
@article{cardoso2020leveraging,
  title={Leveraging the Self-Transition Probability of Ordinal Pattern Transition Graph for Transportation Mode Classification},
  author={Cardoso-Pereira, I and Borges, JB and Barros, PH and Loureiro, AF and Rosso, OA and Ramos, HS},
  journal={arXiv preprint arXiv:2007.08687},
  year={2020}}
```

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


Additionally, it is necessary to download the GeoLife dataset in this [link](https://www.microsoft.com/en-us/download/details.aspx?id=52367). Move the folder called `Data/` into the project folder `db/GeoLife/`.

### Project File Structure

- Please put the `Data/` folder from GeoLife dataset under the folder called `db/GeoLife/`;
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

```
python example.py
```

The running process takes around 2 minutes to finish the whole pipeline of our framework in a little snippet of the GeoLife dataset. All the results will be saved in the folder `db/GeoLife/example/`.

## Support

Feel free to send any questions, comments, or suggestions (in english or portuguese) to Isadora Cardoso-Pereira (isadoracardoso@dcc.ufmg.br).
