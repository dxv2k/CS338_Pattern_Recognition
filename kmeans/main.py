from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt 

# Generate data with 200 samples, 5 random clusters
X, y = make_blobs(n_samples=800, centers=5, random_state=101)
# plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})
# plt.scatter(X[:, 0], X[:, 1])
# # plt.show()
# print(y)






from src.kmeans import * 

cluster_algo = K_means(K = 5)

centroids = cluster_algo._init_centroid(X)
cls = cluster_algo._assign_to_clusters(X,centroids) 
cluster_algo.fit(X)
print(cls)



# from utils.util import *  

# print(get_initial_centroids(X,5))
# print(utils.path)