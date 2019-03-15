import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer

# Define Sigmoid Fuction
def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

# Define Initial Small Number
def iniSmall(dim):
    w = np.random.rand(dim*10).reshape(dim, 10)/100
    b = 0
    return w, b

# Define Sigmoid Propagation
def sigPropagate(w, b, X, Y):
    # Forward Propagation
    count = X.shape[1]
    A = sigmoid(np.dot(w.T, X)+b)
    #cost = -1/count*np.sum(trainY*np.log(A)+(1-trainY)*np.log(1-A))

    # Backward Propagation
    dw = 1/count*np.dot(X, (A-Y).T)
    db = 1/count*np.sum(A-Y)

    grads = {"dw": dw,
             "db": db}

    return grads #, cost

# Define Optimize
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

# Define Prediction
def predict(w, b, X):
    count = X.shape[1]
    Y_prediction = np.zeros((10, count))
    w = w.reshape(X.shape[0], 10)

    A = sigmoid(np.dot(w.T,X)+b)


    for i in range(A.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        if A[:,i] >=0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
        ### END CODE HERE ###

    assert(Y_prediction.shape == (1, count))

    return Y_prediction


# Reading Kaggle Data
train = pd.read_csv('input/train.csv')
#test = pd.read_csv('input/test.csv')

# Encoding
lb_style = LabelBinarizer()
lb_train_results = lb_style.fit_transform(train.iloc[:, 0])
trainY = pd.DataFrame(lb_train_results, columns=lb_style.classes_).head()

# reshape
trainY = lb_train_results.reshape(10, 42000)
trainX = train.iloc[:,1:].values.T

#testX = test.values.T
#testX.shape








w, b = iniSmall(trainX.shape[0])

# Gradient descent
parameters, grads = optimize(w, b, trainX, trainY, numIter=2000, learningRate=0.005)

# Retrieve parameters w and b from dictionary "parameters"
w = parameters["w"]
b = parameters["b"]