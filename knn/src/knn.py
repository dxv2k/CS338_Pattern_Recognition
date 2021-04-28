import numpy as np 
from collections import defaultdict

class KNN:
    def __init__(self, n_neighbors=5, 
                weights='uniform', 
                distance_metric='euclidean'
    ):
        ''' 
        param: 
            n_neighbors: number of neighbors for KNN classifier  
            weights: weight function used in prediction  
                * 'uniform': uniform weights; all points in each neighbor are weighted equally 
                * 'distance': NOT SUPPORTED, currently working on  
            distance_metric: 'euclidean' distance that used for the whole program
        return: 
            None
        ''' 
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.distance_metric = distance_metric
        self.data = None 
        self.label = None

    def fit(self, X, y):
        ''' 
        Feed data set in to K-Nearest Neighbor Algorithm 
        param: 
            X: numpy array contains data points 
            y: numpy array label  
        return: 
            None
        '''
        self.data = X
        self.label = y

    def _distance(self, data1, data2):
        ''' 
        Compute distance based on `self.distance_metric`
        param: 
            data1: numpy array dataa point  
            data2: numpy array dataa point 
        return: 
            matrix distance of data1 and data2 
        '''
        if self.distance_metric == 'manhattan':
            return sum(abs(data1 - data2))          
        elif self.distance_metric == 'euclidean':
            return np.sqrt(sum((data1 - data2)**2))

    def _compute_weights(self, distances):
        ''' 
        Compute weights based on given choice `uniform`, `distance`
        param: 
            distances: receive distances from self._distance() method
        return: 
            array of weights  
        ''' 
        weights = []
        if self.weights == 'uniform':
            for d, y in distances: 
                weights.append([1,y])

        return weights 

    def _predict_one(self, data):
        ''' 
        Predict that given data point is belong or not  
        param: 
            data: numpy array data points to perform prediction
        return: 
            array with len(data) and label per data points 
        ''' 
        distances = sorted((self._distance(x, data), y) 
                        for x, y in zip(self.data, self.label))
        weights = self._compute_weights(distances[:self.n_neighbors])
        weights_by_class = defaultdict(list)

        for d, c in weights:
            weights_by_class[c].append(d)

        counts = [(sum(val), key) for key, val in weights_by_class.items()]
        majority = max(counts)
        return majority[1]

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def score(self, X, y):
        return sum(1 for predict, test in zip(self.predict(X), y) if predict == test) / len(y)


