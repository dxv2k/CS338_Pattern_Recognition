import pandas as pd 
import matplotlib.pyplot as plt

# NOTICE: This K_means only expected numberical dataset 
class K_means: 
    def __init__(self, K = 5, max_iter = 100, tol = 0.001):
        ''' 
        param: 
            K: number of clusters/centroids 
            max_iter: maximum number of itertions 
            tol: tolerance threshold, use to check consecutives iterations 
                                    to declare convergence 
        ''' 
        self.K = K 
        self.max_iter = max_iter
        self.tol = tol

    def _init_centroid(self,data, random_init = False): 
        ''' 
        Initialize centroid randomly or based on first few elements of dataset
        param: 
            data: 
            random_init: True/False 
        ''' 
        centroids = {}
        if random_init: 
            centroids = np.array([]).reshape(n,0)

        # for k in range(self.K): 
            # random init value of centroid
            # centroids = 
        return centroids  


    def train(self, data): 
        ''' 
        param: 
        ''' 
        # init centroid
        self.centroids = {}
        for i in range(self.K): 
            self.centroids[i] = data[i]

        # classifiation
        for i in range(self.max_iter): 
            self.classification = {}

        # iterate till max iter

        pass 

    def predict(self, data): 
        ''' 
        param: 
        ''' 
        classification = None  
        return classification

