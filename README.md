# MSCS_634_Lab2
MSCS_634_Lab2
Overview

This project investigates two supervised classification approaches, K-Nearest Neighbors (KNN) and Radius Neighbors, using the Wine dataset provided by scikit-learn.

The main purpose of the laboratory is to examine how different neighborhood settings influence classification accuracy.

Dataset

The experiment uses the Wine dataset from sklearn.datasets.

The dataset contains:

178 observations
13 numerical predictor variables
3 wine classes

The dataset was divided into training and testing portions using an 80/20 split. Stratification was applied to preserve the class distribution.

Data Preparation

Because both classifiers rely on distance calculations, the numerical predictor variables were standardized before training.

The scaler was fitted using only the training data and then applied to both the training and testing sets.

KNN Experiment

The KNN classifier was evaluated with:

k = 1
k = 5
k = 11
k = 15
k = 21

The accuracy obtained for each configuration was recorded and displayed in the notebook.

Radius Neighbors Experiment

The Radius Neighbors classifier was evaluated with:

Radius = 350
Radius = 400
Radius = 450
Radius = 500
Radius = 550
Radius = 600

The resulting accuracy values were recorded and visualized.

Key Observations

The KNN experiment demonstrates that the number of neighbors affects the classification decision. Smaller values of k make predictions more dependent on nearby individual observations, while larger values incorporate information from a broader neighborhood.

The Radius Neighbors experiment demonstrates a different approach to defining neighborhoods. The number of neighbors can vary from one observation to another because membership is determined by distance.

The specified radius values are relatively large after standardization, so the resulting neighborhoods can contain many training observations. This is an important consideration when interpreting the Radius Neighbors results.

Challenges and Decisions

The primary preprocessing decision was to standardize the predictor variables because both algorithms use distance calculations.

The train-test split was made reproducible with random_state=42. Stratification was also used so that the three target classes would remain represented in both datasets.

Another consideration was the scale of the radius values. The assigned values were retained according to the laboratory requirements, even though they may produce relatively broad neighborhoods after standardization.



