import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer

# Reading Kaggle Data
train = pd.read_csv('input/train.csv')

# Limited the Data Size
# train = train.iloc[:4000,:]

train = train[train['label']<2]



# Brief Overview
train.shape
train.iloc[:, 0]


## Check if Evenly Distributed
train.iloc[:,:2].groupby(['label']).count()
sns.set(style="darkgrid")
sns.countplot(x="label", data=train)
plt.show()

# Encoding
#lb_style = LabelBinarizer()
#lb_results = lb_style.fit_transform(train.iloc[:,0])
#pd.DataFrame(lb_results, columns=lb_style.classes_).head()

# Reshape and Separate
trainY = train.iloc[:, 0].reshape(1, train.shape[0])
trainX = train.iloc[:,1:].values.T


# Useful Info
count = trainX.shape[1]
length = int(np.sqrt(trainX.shape[0]))
height = int(np.sqrt(trainX.shape[0]))

# Print the number
index = 8
plt.imshow(trainX.reshape(height, length, count)[:,:,index])
plt.show()


# Check Shape
trainX.shape
trainY.shape

# Normalize
trainX = trainX/255

# Define Sigmoid Fuction
def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

## check
print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

# Define Initial Small Number
def iniSmall(dim):
    w = np.random.rand(dim).reshape(dim, 1)/100
    b = 0
    return w, b

## Check
w, b = iniSmall(3)
print(w)
print(b)


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros(dim).reshape(dim, 1)
    b = 0
    ### END CODE HERE ###

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


## Check
w, b = initialize_with_zeros(3)
print(w)
print(b)






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

## check
w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads = sigPropagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
#print ("cost = " + str(cost))

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

# check
params, grads = optimize(w, b, X, Y, numIter=100, learningRate=0.009)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

# Define Prediction
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

## check
w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(w, b, X)))





w, b = iniSmall(trainX.shape[0])


# Gradient descent
parameters, grads = optimize(w, b, trainX, trainY, numIter=20000, learningRate=0.005)

# Retrieve parameters w and b from dictionary "parameters"
w = parameters["w"]
b = parameters["b"]



Y_prediction_train = predict(w, b, trainX)


np.sum(trainY == Y_prediction_train)/Y_prediction_train.shape[1]*100


