# K-Nearest-Neighbors-Implementation
Implements the KNN algorithm using cosine distance with functions baked healthily from scratch

nearest_neighbors(train, test, Y_train, n_neighbors=5) does the heavy lifting work in this implementation 

train is a numpy ndarray containing the covariates of the training data

test is a numpy ndarray containing the covaraites of the test data 

Y_train isa numpy array containing the labels of the training data 

n_neighbors is the number of neighbors we want to include in determining the lable of the test data 

Testing on the Iris and red wine quality data sets: 

I tested the algorithm on the iris dataset. First, we shuffle the iris dataset, split the data into 120 training and 30 test items, and then calculate the accuracy level. We do this 20 times over and calculate the average accuracy level for those 20 trials. The average accuracy level that I got was about 97%. The Iris data is an easy test case since the data is closely clustered together

After that I tested the algorithm on a harder case, trying to predict wine quality which takes values of 3, 4, 5, 6, 7, 8 on a scale of 1 (fairly bad) to 10 (very good wine) based on 11 different features. The data set comes of the UC Irvine Machine Learning Data Archives. Again, I calculated the accuracy level for 20 trials with reshuffled data. Here the algorithm has a harder time, producing an accuracy level of around 54%. However, if we allow for some fuzziness in the prediction, giving the algorithm a score when the prediction is +/-1 of the true value, we get an accuracy level of around 91%. 
