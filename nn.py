import numpy as np
import math
from numpy import array


#read iris data from local
def load_iris(file_name):
    file = open(file_name)
    rows = file.read().splitlines() # split the lines at line boundaries. returns a list of lines
    file.close()
    dataset = []
    #for each data row, create a new list to store four features
    for i in range(1,len(rows)): # skip first row(name of features)
        col = rows[i].split(',') # create a list of strings after breaking the given string by ','
        item_features = [] #one list for each item
        # for each column
        for j in range(1, len(col) - 1): # skip first column(id) and last column(name of specie)
            val = float(col[j]); #convert values to float, make sure type are not flexible
            item_features.append(val); #add feature value to item list
        dataset.append(item_features);
    dataset = np.array(dataset) # conversion from 2d list to 2d array
    return dataset


#definition of activation function
#choices are tanh, sigmoid, RELU
#def tanh(x):
#    return np.tanh(x)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

#definition of  loss function
#def crosss_entropy_loss:

#definition of vanilla gradient descent : batch gradient descent with a fixed learning rate
#use back propagation to calculate gradient efficiently
#input: gradients of the loss function w.r.t. parameters
#def vanilla_gradient_descent:


#definition of activation func softmax for output layer:
def softmax(outputlayer):
    w_exp = []
    for i in outputlayer:
        w_exp.append(math.exp(i))
    sum_w_exp = sum(w_exp)
    softmax = []
    for i in w_exp:
        softmax.append(i / sum_w_exp)
    return array(softmax)


#build model
#def build_model():


#predict
#def predict:


print("**predict species of iris")
data = load_iris("Iris.csv")
#model = build_model()
#plot_decision_boundary()
