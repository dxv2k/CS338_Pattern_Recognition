import pandas as pd 
import matplotlib.pyplot as plt


class K_means: 
    def __init__(self, n_clusters = 5, max_iter = 100, tol = 0.001):
        ''' 
            n_clusters: number of clusters/centroids 
            max_iter: maximum number of itertions 
            tol: tolerance threshold, use to check consecutives iterations 
                                    to declare convergence 
        ''' 
        self.K = n_clusters 
        self.max_iter = max_iter
        self.tol = tol

    def _init_centroid(self): 
        ''' 
            random initialize centroid
        ''' 
        centroids = np.array([]).reshape(n,0)
        # for k in range(self.K): 
            # random init value of centroid
            # centroids = 
        return centroids  


    def train(self, data): 
        ''' 
        ''' 
        self.centroids = {}
        # loss = {}
        # init centroids 
        # for iter in range(self.max_iter): 
        #     for k in range(self.K): 
        #         # compute distnace 

        return loss 


    def predict(): 
        classification = None  
        return classification

