## Outlier detection by analysis of cluster transitions
Outlier detection method based on CLOSE.

## Introduction
This repository contains the implementation of an outlier detection method based on maximum stable clustering per timestamp (CLOSE). 

## Getting Started
First create a virtual environment and run following inside it:
```python
pip install .
```
After required libraries are installed go to experiments/show_me_ur_friends. This folder contains all experiments for the comparison of our method with https://link.springer.com/chapter/10.1007/978-981-15-1699-3_8.

In each folder to run experiment with our method, enter:

``` 
python sigma_outlier.py
```
To run an experiment with the method of Tatusch et al. enter:
```
python tau_outlier.py
```
After you executed one of the scripts, the corresponding result will appear in the results-folder.