# Code for *RMS: A ML-based system for ICU Respiratory Monitoring and Resource Planning*

This repository contains the code to reproduce experiments of the associated
manuscript 'RMS: A ML-based system for ICU Respiratory Monitoring and Resource Planning'.

> Note: while the code provided here works with the HiRID-II dataset, we are still working on the public release of this dataset. Once released, we will update this repository accordingly to make sure the findings are fully reproducible.

## Key resources

### HiRID-II data

As part of this work, we will release a significantly revised version of the
HiRID dataset, denoted as HiRID-II, on Physionet. It will be a freely accessible critical care dataset
containing data from more than 55,000 patient admissions to the Department
of Intensive Care Medicine, Bern University Hospital, Switzerland, from 2008
to 2019. The initial version of the dataset, HiRID-I, was released as part of
the journal paper 'Early prediction of circulatory failure in the intensive care
unit using Machine Learning' (known as circEWS).

### Models

The machine learning models to solve individual prediction tasks on the
individual patient-level as well as for resource planning were trained using LightGBM.

### Evaluation metrics

We propose an event-based evaluation metric, similar to the one proposed
in circEWS, which bases the recall on the proportion of
caught events, and the precision based on the proportions of generated
alarms that are correct, for the tasks RMS-RF, RMS-VENT and RMS-REXT.
The extubation failure task (RMS-EF) is evaluated using a
conventional AUPRC (Area under the Precision-Recall Curve) metric, evaluated
at the observed time points of extubation. The resource planning task is
evaluated using MAE of predicted ventilator usage in the future.

### Setup

We assume a Linux installation, typically HPC, with a 'Slurm' cluster
scheduler for dispatching jobs like training or data preprocessing,
across the data batches.

1. Install a conda distribution like Miniconda or Anaconda
2. Clone this repository
3. Update dependencies using Conda as required.

## Download data

1. Get access to the HiRID 2.0 dataset on Physionet, after its release. This includes
   1. Getting a credentialed Physionet account
   2. Submit a usage request to the data owner of the HiRID-II dataset.

2. Once access is granted, download the merged stage of the data, from which
   all derived resources in this project can be built.

## Code components

The code is organized in several sub-directories in the Python module
**`RMS`**, which contain the following contents:

* **`endpoints`**
Annotation of time series with respiratory system related annotations.

* **`evaluation`** 
Evaluation of RMS tasks performance and evaluation of resource planning.

* **`exp_design`**  
Code concerned with splitting PIDs for cluster processing and generating data
splits for the experimental design.

* **`imputation`**  
Code concerned with transforming HIRID-II data to a fixed time grid, making it suitable for 
feature generation and fitting of machine learning models. Data is partially imputed
and sometimes left as missing.

* **`introspection`**  
Code concerned with SHAP value analysis and analysis of variable importance for predicting
various RMS tasks.

* **`labels`**  
Code for creating machine learning labels of the various RMS tasks.

* **`learning`**  
Supervised learning scripts for learning risk scores for predicting respiratory failure
as well as other RMS tasks.

* **`ml_dset`**  
Save features/labels in a compact HDF5 format for the training/validation sets.

* **`ml_input`**  
Contains code for generation of features on partially imputed data, the machine learning
labels are also appended to this dataset.

* **`statistics`**  
Various scripts collecting statistics about different stages of the pipeline.

* **`utils`**
Various utility functions used in other modules.


The order in which each component is run is as follows:
1. exp_design
2. imputation 
3. endpoints
4. labels
5. ml_input
6. ml_dset
7. learning
8. evaluation/introspection/statistics

# License

The research code associated with the manuscript is licensed under
a MIT license. The HiRID-II data is licensed as specified on
Physionet.

When using code from this repository, please consider citing

> Hüser, Lyu, Faltys, Pace et al. "A comprehensive ML-based respiratory monitoring system for physiological monitoring & resource planning in the ICU", medRxiv 2024.01.23.24301516


