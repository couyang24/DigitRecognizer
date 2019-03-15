import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

import seaborn as sns

train = pd.read_csv('input/train.csv')

train.shape
train.iloc[0,:]



train.iloc[:,:2].groupby(['label']).count()


sns.set(style="darkgrid")
sns.countplot(x="label", data=train)
plt.show()

trainY = train.iloc[:,0].values.reshape(1, 42000)
trainX = train.iloc[:,1:].values.T


count = trainX.shape[1]
length = int(np.sqrt(trainX.shape[0]))
height = int(np.sqrt(trainX.shape[0]))

index = 3
plt.imshow(trainX.reshape(height, length, count)[:,:,index])
plt.show()

trainX.shape
trainY.shape
trainX = trainX/255

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def iniWB(dim):
    w = np.random.rand(dim).reshape(dim, 1)/100
    b = 0
    return w, b


w, b = iniWB(3)

print(w)
print(b)