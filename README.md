# Naive-Bayes-Classifier-IRIS



This project implements a **Naive Bayes Classifier** to classify the **Iris flower dataset** using Gaussian Naive Bayes principles. The classifier computes likelihood, prior, and posterior probabilities to make predictions and evaluate accuracy.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Implementation Steps](#implementation-steps)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview
The Naive Bayes Classifier is a probabilistic machine learning model based on Bayes' theorem. This project specifically implements a **Gaussian Naive Bayes Classifier**, assuming that each feature follows a normal (Gaussian) distribution.

## Dataset
The **Iris dataset** contains 150 samples of iris flowers from three species (**Setosa, Versicolor, and Virginica**), with four features:
- **Sepal length**
- **Sepal width**
- **Petal length**
- **Petal width**

## Implementation Steps
1. **Cross-Validation & Data Splitting**: 
   - Load the Iris dataset.
   - Split it into training and testing sets (80/20 split) with stratified sampling.

2. **Likelihood Calculation**:  
   - Compute the **mean** and **standard deviation** for each feature in each class.
   - Define a **Gaussian likelihood function** to model the probability distribution.

3. **Prior Probability Calculation**:  
   - Compute the prior probabilities for each class based on their frequency in the training set.

4. **Posterior Probability Calculation**:  
   - Compute posterior probabilities using **Bayes’ theorem** with the likelihood and prior probabilities.

5. **Prediction & Accuracy Evaluation**:  
   - Predict class labels for test samples.
   - Evaluate model performance using **accuracy score**.

6. **Synthetic Sample Generation**:  
   - Generate new samples based on learned Gaussian distributions.

## Installation
To run this project, install the required dependencies:

```bash
pip install numpy scikit-learn

# run the classifier script
python naive_bayes_iris.py

Sample 1 (Class: setosa): [5.13, 3.16, 1.46, 0.25]
Sample 2 (Class: versicolor): [6.42, 3.41, 4.35, 1.34]
Sample 3 (Class: virginica): [7.43, 3.08, 5.98, 1.96]
...



