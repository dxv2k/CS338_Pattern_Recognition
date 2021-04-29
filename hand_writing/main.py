# Ref: https://www.kaggle.com/ngbolin/mnist-dataset-digit-recognizer

import pandas as pd
import numpy as np 

data_train = pd.read_csv(r"./data/train_new.csv") 
data_test = pd.read_csv(r"./data/test_new.csv") 



y_train = data_train["label"]

# Drop 'label' column
X_train = data_train.drop(labels = ["label"],axis = 1) 

# free some space
del data_train 

print(y_train.head())

