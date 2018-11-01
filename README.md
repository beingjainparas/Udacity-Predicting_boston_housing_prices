# Project Overview: Overview
The Boston housing market is highly competitive, and we want to be the best real estate agent in the area. To compete with our peers, we decide to leverage a few basic machine learning concepts to assist us and a client with finding the best selling price for their home. Luckily, we’ve come across the Boston Housing dataset which contains aggregated data on various features for houses in Greater Boston communities, including the median value of homes for each of those areas. Our task is to build an optimal model based on a statistical analysis with the tools available. This model will then be used to estimate the best selling price for our clients' homes.

## What did we need to install?
We needed an installation of Python, plus the following libraries:
* Pandas
* NumPy
* Matplotlib
* scikit-learn

>We recommend installing [Anaconda](https://www.continuum.io/downloads), which comes with all of the necessary packages, as well as IPython notebook.

## Project Highlights
This project is designed to get us acquainted with the many techniques for training, testing, evaluating, and optimizing models, available in sklearn also working with datasets in Python and applying basic machine learning techniques using NumPy and Scikit-Learn. Before being expected to use many of the available algorithms in the sklearn library, it will be helpful to first practice analyzing and interpreting the performance of our model.

Things we will learn by completing this project:
* How to use NumPy to investigate the latent features of a dataset.
* How to analyze various learning performance plots for variance and bias.
* How to determine the best-guess model for predictions from unseen data.
* How to evaluate a model's performance on unseen data using previous data.
* How to explore data and observe features.
* How to train and test models.
* How to identify potential problems, such as errors due to bias or variance.
* How to apply techniques to improve the model, such as cross-validation and grid search.

# Project Description: Description
In this project, we will apply basic machine learning concepts on data collected for housing prices in the Boston, Massachusetts area to predict the selling price of a new home. We will first explore the data to obtain important features and descriptive statistics about the dataset. Next, we will properly split the data into testing and training subsets, and determine a suitable performance metric for this problem. We will then analyze performance graphs for a learning algorithm with varying parameters and training set sizes. This will enable us to pick the optimal model that best generalizes for unseen data. Finally, we will test this optimal model on a new sample and compare the predicted selling price to our statistics.

# Project Details: Starting the Project
This project contains four files:

* [boston_housing.ipynb](https://github.com/beingjainparas/Udacity-Predicting_boston_housing_prices/blob/master/boston_housing.ipynb): This is the main file where we have performed our work on the project.
* [housing.csv](https://github.com/beingjainparas/Udacity-Predicting_boston_housing_prices/blob/master/housing.csv): The project dataset. We'll load this data in the notebook.
* [visuals.py](https://github.com/beingjainparas/Udacity-Predicting_boston_housing_prices/blob/master/visuals.py): This Python script provides supplementary visualizations for the project. Do not modify.
* `report.html`: An HTML export of the project notebook with the name report.html.

In the Terminal or Command Prompt, navigate to the folder containing the project files, and then use the command `jupyter notebook boston_housing.ipynb` to open up a browser window or tab to work with our notebook. Alternatively, we can use the command `jupyter notebook` or `ipython notebook` and navigate to the notebook file in the browser window that opens.

## Dataset
The modified Boston housing dataset consists of 489 data points, with each datapoint having 3 features. This dataset is a modified version of the Boston Housing dataset found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing).

**Features**
* `RM`: average number of rooms per dwelling
* `LSTAT`: percentage of population considered lower status
* `PTRATIO`: pupil-teacher ratio by town

**Target Variable**
* `MEDV`: median value of owner-occupied homes

# Project Rubics
[Check Here](https://review.udacity.com/#!/rubrics/103/view)

# Extra Links that halped me solve the project are:
* https://morphocode.com/pandas-cheat-sheet/
* http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
