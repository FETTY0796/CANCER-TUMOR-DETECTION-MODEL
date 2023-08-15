#!/usr/bin/env python
# coding: utf-8

# In[40]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
data


# In[41]:


# Assume that the first column is the ID, second is the label (M = malignant, B = benign), and the rest are features
X = data.loc[:, 2:].values
y = data.loc[:, 1].values


# In[42]:


# Convert labels to integers for the model
y = (y == 'M').astype(int)


# In[43]:


# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[44]:


# Scale the features for better performance
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[45]:


# Create a logistic regression model
model = LogisticRegression()


# In[46]:


# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)


# In[47]:


# Print metrics to evaluate the model
print(f'Accuracy: {accuracy_score(y_test, predictions)}')
print(f'Confusion Matrix: \n {confusion_matrix(y_test, predictions)}')
print(f'Classification Report: \n {classification_report(y_test, predictions)}')


# In[50]:


# Import necessary libraries
import numpy as np

# Here is an example of a new data sample (this should have the same number of features as your training data)
new_sample = np.array([1.79900000e+01, 1.03800000e+01, 1.22800000e+02, 1.00100000e+03,
       1.18400000e-01, 2.77600000e-01, 3.00100000e-01, 1.47100000e-01,
       2.41900000e-01, 4.87100000e-02, 1.09500000e+00, 9.05300000e-01,
       8.58900000e+00, 1.53400000e+02, 6.39900000e-03, 4.90400000e-02,
       5.37300000e-02, 1.58700000e-02, 3.00300000e-02, 6.19300000e-03,
       2.53800000e+01, 1.73300000e+01, 1.84600000e+02, 2.01900000e+03,
       1.62200000e-01, 6.65600000e-01, 7.11900000e-01, 2.65400000e-01,
       4.60100000e-01, 1.18900000e-01])

# Note: this new_sample is an example and the actual sample should be collected from the field or given in your problem statement.

# Reshape the new sample to match the input shape that the model expects
new_sample = new_sample.reshape(1, -1)

# Scale the new sample using the same scaler used for the training data
new_sample = sc.transform(new_sample)

# Use the model to make a prediction for the new sample
new_prediction = model.predict(new_sample)

# Print out the prediction
if new_prediction[0] == 0:
    print("The model predicts that the tumor is benign.")
else:
    print("The model predicts that the tumor is malignant.")


# In[ ]:




