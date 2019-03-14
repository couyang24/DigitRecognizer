import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage

import seaborn as sns

train = pd.read_csv('input/train.csv')

train.shape
train.iloc[0,:]

train.shape

train = train.iloc[:4000,:]

train.iloc[:,:2].groupby(['label']).count()


sns.set(style="darkgrid")
sns.countplot(x="label", data=train)
plt.show()



from sklearn.preprocessing import LabelBinarizer

lb_style = LabelBinarizer()
lb_results = lb_style.fit_transform(train.iloc[:,0])
pd.DataFrame(lb_results, columns=lb_style.classes_).head()


trainY = lb_results.reshape(10, 4000)
trainX = train.iloc[:,1:].values.T


count = trainX.shape[1]
length = int(np.sqrt(trainX.shape[0]))
height = int(np.sqrt(trainX.shape[0]))

index = 8
plt.imshow(trainX.reshape(height, length, count)[:,:,index])
plt.show()

trainX.shape
trainY.shape
trainX = trainX/255









def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s


print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

def iniSmall(dim):
    w = np.random.rand(dim).reshape(dim, 1)/100
    b = 0
    return w, b


w, b = iniSmall(3)

print(w)
print(b)

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



def predict(w, b, X):
    count = X.shape[1]
    Y_prediction = np.zeros((1, count))
    w = w.reshape(X.shape[0], 1)

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


w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(w, b, X)))



trainY[0,:].shape
trainY[0,:].reshape(1,4000)



w, b = iniSmall(trainX.shape[0])


trainX.shape
trainY[0,:].reshape(4000,1).shape


# Gradient descent (≈ 1 line of code)
parameters, grads = optimize(w, b, trainX, trainY[0,:].reshape(1,4000), numIter=200, learningRate=0.005)

# Retrieve parameters w and b from dictionary "parameters"
w = parameters["w"]
b = parameters["b"]



Y_prediction_train = predict(w, b, trainX)


Y_prediction_train.shape
Y_prediction_train[:,40:50]

trainY[0,:].reshape(1,4000)