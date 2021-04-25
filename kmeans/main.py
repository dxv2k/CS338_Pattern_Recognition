# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use('ggplot')
# import numpy as np

# X = np.array([[1, 2],
#               [1.5, 1.8],
#               [5, 8 ],
#               [8, 8],
#               [1, 0.6],
#               [9,11]])

# plt.scatter(X[:,0], X[:,1], s=150)
# plt.show()

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt 

# Generate data with 200 samples, 5 random clusters
X, y = make_blobs(n_samples=800, centers=5, random_state=101)
# plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()
# print(y)

from src.kmeans import kmeans 

clf = kmeans.K_means()

from utils.util import *  

print(get_initial_centroids(X,5))
# print(utils.path)