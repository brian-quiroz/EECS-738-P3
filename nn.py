import numpy as np
import math
from numpy import array
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

#    set regularization strength to 0.01
regularization_strength = 0.001

#read iris data from local
def load_iris(file_name):
    file = open(file_name)
    rows = file.read().splitlines() # split the lines at line boundaries. returns a list of lines
    file.close()
    dataset = []
    labels = []
    #for each data row, create a new list to store four features
    for i in range(1,len(rows)): # skip first row(name of features)
        col = rows[i].split(',') # create a list of strings after breaking the given string by ','
        item_features = [] #one list for each item
        for j in range(1, len(col)): # skip first column(id)
            if j == len(col) - 1: #add last column(name of species) to labels
                if col[j] == "Iris-setosa":
                    labels.append(0)
                elif col[j] == "Iris-versicolor":
                    labels.append(1)
                elif col[j] == "Iris-virginica":
                    labels.append(2)
            else:
                val = float(col[j]); #convert values to float
                item_features.append(val); #add feature value to item list
        dataset.append(item_features);
    dataset = np.array(dataset) # conversion from 2d list to 2d array
    labels = np.array(labels)
    return dataset, labels


#definition of activation function
def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def softmax(output):
    exp_scores = np.exp(output)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

#definition of  loss function
def calculate_loss(model, X, y, num_examples):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    #get predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    probs = softmax(z2)

    # cross-entropy loss
    crosss_entropy_loss = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(crosss_entropy_loss)

    return 1./num_examples * data_loss




'''build model
    input:
    dim_of_hl, dimension of hidden layer
    dim_of_il, dimension of input layer
    dim_of_ol, dimension of output layer
    X, input data
    y, expected result
    epilon
    output:
    model, with four parameters
'''
def build_model(dim_of_hl, dim_of_il, dim_of_ol, X, y, epsilon, num_examples,iterations = 30000, print_loss = True):

    #initialize weights and offset
    np.random.seed(0)#make sure get same set of random numbers
    W1 = np.random.randn(dim_of_il, dim_of_hl)/ np.sqrt (dim_of_il) #return W1 samples from standard normal distribution
    b1 = np.zeros((1,dim_of_hl))
    W2 = np.random.randn(dim_of_hl, dim_of_ol)/ np.sqrt (dim_of_hl)
    b2 = np.zeros((1,dim_of_ol))
    model = {}
    for i in range(0, iterations):
        #feed forward
        z1 = X.dot(W1) + b1
        a1 = tanh(z1)
        z2 = a1.dot(W2) + b2
        probs = softmax(z2)

        #back propagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        #regularization
        dW2 += regularization_strength * W2
        dW1 += regularization_strength * W1

        # batch gradient descent
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # update model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        #print total loss every 1000 iterations to track accuracy
        if print_loss and i % 1000 == 0:
            loss = calculate_loss(model, X, y, num_examples)
            print("total loss after iteration %i: %f" % (i, loss))
    return model



#predict
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    probs = softmax(z2)
    return np.argmax(probs, axis=1)



print("**predict species of iris")
data, labels = load_iris("Iris.csv")
'''
    number of species of iris is 3, so dim_of_il and dim_of_ol are both 3.
    based on Kolmogorov theorem, set dim_of_hl to 2 * dime_of_il + 1
    set epsilon (learning rate) to 0.01
    '''
num_examples = len(data)
model = build_model(15, 4, 3, data, labels, 0.001 ,num_examples) #when learning rate == 0.001, it reaches decent results
