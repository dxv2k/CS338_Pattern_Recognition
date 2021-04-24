import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# NOTICE: This K_means only expected numberical dataset 
# Steps: 
# 1. Random selection of centroids 
# 2. Calculate distance to each point and assign each point to cluster 
# 3. Calculate AVG of the assigned points and move 
#    centroids to new position 
# Repate 2-3 until convergence



# Ref: https://github.com/tugot17/K-Means-Algorithm-From-Scratch/blob/master/k-means.py

def euclidean_distance(A_matrix, B_matrix): 
    ''' 
    Function computes Euclidean distance for every elements of A to B
    E.g: C[2,15] is distance between point 2 from A[2] matrix and point 15 from matrix B[15]
    param: 
        A_matrix: numpy array type, N1:D 
        B_matrix: numpy array type, N2:D
    return: 
        distance: numpy array dtype, N1:N2
    ''' 
        distance = numpy.linalg.norm(A_matrix - B_matrix)
    return distance

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

    # TODO: complete _init_centroids 
    def _init_centroid(self,data, random_init = False): 
        ''' 
        Initialize centroid randomly or based on first few elements of dataset
        param: 
            data: data use to train K-means algorithm 
            random_init: True/False 
        return: 
            centroids: dictionary datatype, size of K
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
        pass 