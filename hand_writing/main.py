# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.layers.kernelized import RandomFourierFeatures 
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from tensorflow.keras.layers import InputLayer

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np

np.random.seed(1212)

import keras
from keras.models import Model
from keras.layers import *
from keras import optimizers


train = pd.read_csv(r"../hand_writing_data/train_new.csv")
test = pd.read_csv(r"../hand_writing_data/test_new.csv")
submission = pd.read_csv(r"../hand_writing_data/sample_submission_new.csv")


# class CNN: 
#     def __init__(self):
#         self.model = None 

#     # Build architecture of the network
#     # def build(self, input_dim, output_dim):
#     def build(self):
#         # Input Parameters
#         n_input = 784 # number of features
#         n_hidden_1 = 300
#         n_hidden_2 = 100
#         n_hidden_3 = 100
#         n_hidden_4 = 200
#         num_digits = 10        
#         Inp = tf.keras.Input(shape=(784,))
#         x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)
#         x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)
#         x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)
#         x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)
#         output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)

#         self.model = Model(Inp, output)


#     # def train(self): 

#     # def predict(self):

#     # def evaluate(self):

#     def summary(self): 
#         return model.summary()



# model = CNN()
# # model.build(28,10)
# model.build()




# Input Parameters
n_input = 784 # number of features
n_hidden_1 = 300
n_hidden_2 = 100
n_hidden_3 = 100
n_hidden_4 = 200
num_digits = 10

Inp = Input(shape=(784,))
x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)
x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)
x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)
x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)
output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)


# Our model would have '6' layers - input layer, 4 hidden layer and 1 output layer
model = Model(Inp, output)
model.summary() # We have 297,910 parameters to estimate

# Ref: https://www.kaggle.com/ngbolin/mnist-dataset-digit-recognizer


