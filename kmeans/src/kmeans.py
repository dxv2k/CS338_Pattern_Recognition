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
    E.g: C[2,15] is distance between point 2 from A[2] matrix 
        and point 15 from matrix B[15]
    param: 
        A_matrix: numpy array type, N1:D 
        B_matrix: numpy array type, N2:D
    return: 
        distance: numpy array dtype, N1:N2
    ''' 
    A_square = np.reshape(np.sum(A_matrix * A_matrix, axis=1), (A_matrix.shape[0], 1))
    B_square = np.reshape(np.sum(B_matrix * B_matrix, axis=1), (1, B_matrix.shape[0]))
    AB = np.dot(A_matrix,B_matrix.T) 
    C = -2 * AB + B_square + A_square
    return np.sqrt(C)


class K_means: 
    def __init__(self, K = 5, max_iter = 100, 
                tol = 0.001, 
                distance_measuring_method = euclidean_distance):
        ''' 
        param: 
            K: number of clusters/centroids 
            max_iter: maximum number of itertions 
            tol: tolerance threshold, use to check consecutives iterations 
                                    to declare convergence 
            distance_measure_method: default is Euclidean distance method 
        ''' 
        self.K = K 
        self.max_iter = max_iter
        self.tol = tol
        self.distance_method = distance_measuring_method
        self.centroids = {} # number of centroids = number of K 
                            # which stores data points 

    # TODO: complete _init_centroids with randomly selection 
    def _init_centroid(self,data): 
        ''' 
        Initialize centroid from the data points based on number K 
        param: 
            data: data use to train K-means algorithm 
        return: 
            centroids: dictionary datatype, size of K
        ''' 
        centroids = {}
        for k in range(self.K): 
            self.centroids[k] = data[k]

        return centroids  
    
    def _has_convergence(self,prev_centroids, curr_centroids): 
        ''' 
        Check if any centroids moved more than 'self.tol'/tolerance threshold 
        Default measurement distance is Euclidean distance
        param: 
            prev_centroids: numpy array
            curr_centroids
            distance_measuring_method
        return: 
            True if centroids is covered, False if not 
        '''
        distance_between_centroids = self.distance_method(prev_centroids, curr_centroids) 
        # TODO: why checking diagonal of the distance matrix? 
        # pseudo code: if distance_matrix.diagonal() >= thresholc -> True, else -> False
        centroids_covered = np.max(distance_between_centroids.diagonal()) <= self.tol
        return centroids_covered

    # TODO: 
    def _assign_to_clusters(self,data, centroids): 
        ''' 
        Assign N data points to clusters based on given centroids
        param: 
            data: numpy array  
            centroids
        return: 
            dict type {cluster_number: list of points in cluster}
        '''
        clusters = {}
        # init empty dictionray with size of K 
        for i in range(self.K): 
            clusters[i] = []

        # compute distance of each data points to centroids  
        distance = self.distance_method(data, centroids)

        # TODO: no understanding why using argmin result in indices
        # Doc: https://numpy.org/doc/stable/reference/generated/numpy.indices.html
        # Doc: https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
        closest_cluster_ids = np.argmin(distance, axis = 1)
        for idx, cluster_id in enumerate(closest_cluster_ids):
            clusters[cluster_id].append(data[i])

        return clusters

    # TODO: complete train function
    def train(self, data): 
        ''' 
        Perform K-means clusterings on given dataset
        param: 
            data  
        return: 

        ''' 

        # init centroid
        new_centroids = self._init_centroid(data = data)
        
        centroids_convered = False 
        # main training loop
        while not centroids_convered: 
            prev_centroids = new_centroids 
            clusters = self._assign_to_clusters(data,prev_centroids)
            # TODO: working on compute new_centroids compute average 
            centroids_convered = self._has_convergence(prev_centroids, new_centroids)


        self.centroid = new_centroids
        
        return self.centroids 

    def predict(self, data): 
        ''' 
        param: 
        return: 
        ''' 
        pass 

