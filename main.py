import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
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



del train



def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def iniSmall(dim):
    w = np.random.rand(dim).reshape(dim, 1)/100
    b = 0
    return w, b


w, b = iniSmall(3)

print(w)
print(b)

def sigPropagate(w, b, trainX, trainY):
    # Forward Propagation
    count = trainX.shape[1]
    A = sigmoid(np.dot(w.T, trainX)+b)
    #cost = -1/count*np.sum(trainY*np.log(A)+(1-trainY)*np.log(1-A))

    # Backward Propagation
    dw = 1/count*np.dot(trainX, (A-trainY).T)
    db = 1/count*np.sum(A-trainY)

    grads = {"dw": dw,
             "db": db}

    return grads #, cost

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads = sigPropagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
#print ("cost = " + str(cost))




def optimize(w, b, X, Y, numIter, learningRate):

    for index in range(numIter):
        grads = sigPropagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w -= learningRate*dw
        b -= learningRate*db

    params = {"w": w,
                  "b": b}
    grads = {"dw": dw,
                 "db": db}

    return params, grads


params, grads = optimize(w, b, X, Y, numIter=100, learningRate=0.009)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))







