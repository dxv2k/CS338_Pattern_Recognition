from src.knn import KNN
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np 


# Iris Dataset 
iris = datasets.load_iris()
X_train, X_temp, y_train, y_temp = \
    train_test_split(iris.data, iris.target, test_size=.3)
X_validation, X_test, y_validation, y_test = \
    train_test_split(X_temp, y_temp, test_size=.3)

# Initialize KNN algorithm
neighbor = KNN()

# Feed dataset into KNN
neighbor.fit(X_train,y_train)

# Evaluate prediction of KNN 
print(neighbor.predict(X_train))
print(neighbor.score(X_train, y_train))
print(neighbor.score(X_test, y_test))

#####################################################################
# Multi-class classification Toy example 
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [8, 8]])
y = np.array([1,1,1,0,0,2])

neighbor.fit(X,y)
print(neighbor.predict(X))
print(neighbor.score(X,y))