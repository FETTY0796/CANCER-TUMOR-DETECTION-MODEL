# CANCER-TUMOR-DETECTION-MODEL


#Breast Cancer Detection Model
Overview
This repository contains a breast cancer detection model based on the breast cancer Wisconsin dataset. The primary objective is to classify a tumor as benign (B) or malignant (M) based on the features extracted from a digitized image of a fine needle aspirate (FNA) of a breast mass.

Dataset
The dataset is sourced from the UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic) dataset.

The dataset consists of 30 features, including mean radius, mean texture, mean perimeter, and mean area. The target variable is binary: M (malignant) or B (benign).

##Setup and Requirements

Libraries:
pandas
sklearn
numpy
To run the model, ensure you have the above libraries installed. You can usually install these with pip:

Copy code
pip install pandas scikit-learn numpy


##Implementation Details
The dataset is loaded directly from the UCI repository using pandas.
Features and labels are extracted, and labels are encoded as integers (M as 1 and B as 0).
The data is split into training and test sets.
Features are scaled using StandardScaler from sklearn.preprocessing for better performance.
A logistic regression model from sklearn.linear_model is trained on the training set.
The model's performance is evaluated on the test set using accuracy, confusion matrix, and a classification report.
An example of predicting a new data sample is provided, where the sample is processed and classified using the trained model.
Usage
Run the provided Python script. At the end of the script, an example prediction is made on a sample data point to showcase the model's prediction capability.

##Contribution
For anyone looking to contribute, ensure that:

Any enhancements or bug fixes made do not compromise the accuracy of the model.
Adhere to PEP 8 coding standards.
Update the README if necessary, with clear details of changes made.

##License
This project is open source. Feel free to use, modify, and distribute the code, but always acknowledge the source.

