# -*- coding: utf-8 -*-
"""
Created on Tue May  2 09:15:47 2017

@author: nbui151
"""

import numpy as np
import pandas as pd
import operator
from collections import defaultdict

# Function to calculate cosine similarity
def Cos(u, v):
    cosine = np.inner(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
    return cosine 

'''
Function to find nearest neighbors and predict the label of Y 
train is a numpy ndarray containing the covariates of the training data
test is a numpy ndarray containing the covaraites of the test data 
Y_train isa numpy array containing the labels of the training data 
n_neighbors is the number of neighbors we want to include in determining the lable of the test data 
''' 

def nearest_neighbors(train, test, Y_train, n_neighbors=5):
    predict = [] # stores the predicted labels 
    for m in range(test.shape[0]): # loop through the test data 
        
        v = test[m] 
        # calculate distance between vector v and all the vectors in the training data set
        distance = {}
        for i in range(train.shape[0]):
            u = train[i]
            distance_metric = Cos(u, v)
            distance[i] = distance_metric
            
        # distance_sort gives the sorted distance between v and all training items 
        distance_sort = sorted(distance.items(), key=operator.itemgetter(1), reverse=True)
        
        # extract the ID's of the closest neighbors 
        neighbors_id = []
        for i in range(n_neighbors):
            if n_neighbors < len(distance_sort):
                neighbors_id.append(distance_sort[i][0])
        
        # extract the labels of the closest neighbors  
        neighbors_target = []
        for item in neighbors_id:
            neighbors_target.append(Y_train[item])
        target_dict = defaultdict(int)
        for i in neighbors_target:
            target_dict[i] += 1 
        
        # sort the labels to find the most frequent one 
        target_dict_sort = sorted(target_dict.items(), key=operator.itemgetter(1), reverse=True)
        predict_target = target_dict_sort[0][0]
        
        # append the label to the predicted labels list 
        predict.append(predict_target)
        
    return predict 

'''
We then test the algorithm on the iris dataset, we shuffle 
the iris dataset, split the data into 120 training and 30 test items,
and calculate the accuracy level. We do this 20 times over and calculate
the average accuracy level for those 20 trials
'''

# Import the Iris data set 
from sklearn import datasets
data_iris = datasets.load_iris()
data_iris_combined = pd.DataFrame(data_iris.data)
data_iris_combined['target'] = data_iris.target

# accuracy_list stores the accuracy level for each trial 
accuracy_list = []
for i in range(20):
    # shuffle the data 
    df = data_iris_combined.sample(frac=1).reset_index(drop=True)
    Y_iris = np.array(df.target) 
    X_iris = np.array(df.drop('target', axis = 1)) 
    # divide into training and test sets 
    train_iris = X_iris[0:120]
    Y_train_iris = Y_iris[0:120]
    test_iris = X_iris[120:151]
    Y_iris_true = Y_iris[120:151]
    # call the nearest_neighbors function 
    Y_iris_predict = nearest_neighbors(train_iris, test_iris, Y_train_iris, n_neighbors = 5)
    accuracy_iris = np.mean(Y_iris_predict == Y_iris_true) *100
    accuracy_list.append(accuracy_iris)

# print out results 
print("Average accuracy for iris dataset is ", np.mean(accuracy_list))

'''
The average accuracy level that I got was about 97%
The Iris data is an easy test case since the data is closely clustered together
Let's test the algorithm on a harder case, predicting wine quality which takes 
values of 3, 4, 5, 6, 7, 8 on a scale of 1 (fairly bad) to 10 (very good wine)
based on 11 different features. The data set comes of the UC Irvine Machine Learning 
Data Archives. Again, we do this for 20 trials with reshuffled data. 
Here the algorithm has a harder time, producing an accuracy level of around 54%. 
However, if we allow for some fuzziness in the prediction, giving the algorithm a 
score when the prediction is +-1 of the true value, we get an accuracy level of around 91%. 
'''

data_wine_original = pd.read_csv("redwinequality.csv")

# define train and test set 
accuracy_list_wine = [] 
accuracy_list_wine_fuzzy = []

for i in range(20): 
    data_wine = data_wine_original.sample(frac=1).reset_index(drop=True)

    Y_wine = np.array(data_wine.quality)
    X_wine = np.array(data_wine.drop('quality', axis = 1))
    train_wine = X_wine[0:1500]
    Y_train_wine = Y_wine[0:1500]
    test_wine = X_wine[1500:1600]
    Y_wine_true = Y_wine[1500:1600]
    Y_wine_predict = nearest_neighbors(train_wine, test_wine, Y_train_wine, n_neighbors = 5)
    
    accuracy_wine = np.mean(Y_wine_predict==Y_wine_true) * 100
    accuracy_list_wine.append(accuracy_wine)
    
    accuracy_fuzzy = np.mean(abs(Y_wine_predict-Y_wine_true)<2) * 100
    accuracy_list_wine_fuzzy.append(accuracy_fuzzy)
            
print("Average strict accuracy for red wine dataset is ", np.mean(accuracy_list_wine))
print("Average fuzzy accuracy for red wine dataset is ", np.mean(accuracy_list_wine_fuzzy))
