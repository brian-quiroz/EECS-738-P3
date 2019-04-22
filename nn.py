import numpy as np
import math
import random
from numpy import array

#read iris data from local
def load_iris(file_name):
    file = open(file_name)
    rows = file.read().splitlines() # split the lines at line boundaries. returns a list of lines
    file.close()
    dataset = []
    speciesDict = {}
    k = 1
    #for each data row, create a new list to store four features
    for i in range(1,len(rows)): # skip first row(name of features)
        col = rows[i].split(',') # create a list of strings after breaking the given string by ','
        item_features = [] #one list for each item
        # for each column
        for j in range(1, len(col) - 1): # skip first column(id) and last column(name of specie)
            val = float(col[j]); #convert values to float, make sure type are not flexible
            item_features.append(val); #add feature value to item list
        species = col[len(col) - 1]
        if species in speciesDict:
            item_features.append(speciesDict[species])
        else:
            speciesDict[species] = k
            item_features.append(speciesDict[species])
            k += 1
        dataset.append(item_features);
    dataset = np.array(dataset) # conversion from 2d list to 2d array
    return dataset


#definition of activation function
#choices are tanh, sigmoid, RELU
#def tanh(x):
#    return np.tanh(x)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoidDerivative(x):
    return sigmoid(x) * (1- sigmoid(x))

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

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        # print("BIASES")
        # for arr in self.biases:
        #     print(1, arr)
        # print("WEIGHTS")
        # for arr in self.weights:
        #     print(2, arr)

    def feedforwardTest(self, a):
        print("BIASES")
        for arr in self.biases:
            print(1, arr)
        print("WEIGHTS")
        for arr in self.weights:
            print(2, arr)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def feedforward(self, x, y):
        activ = x
        activs = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            # print("w", w)
            # print("active", activ)
            # print("b",b)
            z = np.dot(w, activ) + b
            # print("z", z)
            zs.append(z)
            activ = sigmoid(z)
            activs.append(activ)
            # print("\n\n")

        return (zs, activs)

    def backwardPass(self, zs, activs, y):
        bPartial = [np.zeros(b.shape) for b in self.biases]
        wPartial = [np.zeros(w.shape) for w in self.weights]

        delta_L = self.costGradient(activs[-1], y)
        delta_L *= sigmoidDerivative(zs[-1])
        bPartial[-1] = delta_L
        wPartial[-1] = np.dot(delta_L, activs[-2].transpose())

        delta_l = delta_L
        # print("Here")
        for i in range(2, self.num_layers):
            l = self.num_layers - 1 - i
            z = zs[-i]
            sigmoidPrime_of_z = sigmoidDerivative(z)
            # print("ADSFADF", l + 1, -i + 1)
            delta_l = np.dot(self.weights[-i + 1].transpose(), delta_l) * sigmoidPrime_of_z
            bPartial[-i] = delta_l
            wPartial[-i] = np.dot(delta_l, activs[-i - 1].transpose())
            # print("Here2")
            # print(l, ": ", wPartial[l])

        return (bPartial, wPartial)

    #build model
    def build_model(self, training_data, eta):
        # num_examples = len(data)
        # epsilon = 0.01 # learning rate for gradient descent

        n = len(training_data)

        #update every time
        for x, y in training_data:
            zs, activs = self.feedforward(x,y)
            # print("ACTIV_OUT", activs[-1])
            (bPartial, wPartial) = self.backwardPass(zs, activs, y)

            # print("A")
            # for i in self.weights:
            #     print (i,",")
            # print('\n\n')
            # print("B")
            # for i in wPartial:
            #     print (i, ",")
            # print('\n\n')

            self.weights = [w - (eta/n) * nw for w, nw in zip(self.weights, wPartial)]
            self.biases = [b - (eta/n) * nb for b, nb in zip(self.biases, bPartial)]

    def costGradient(self, activL, y):
        return (activL - y)

    def predict(self, testing_data):
        n_test = len(testing_data)
        fft = [self.feedforwardTest(x) for (x,y) in testing_data]
        i = 0
        for f in fft:
            print(i, f, '\n', np.argmax(f))
            i +=1
            print('\n\n\n')
        # print(fft)
        test_results = [(np.argmax(self.feedforwardTest(x)), y) for (x, y) in testing_data]
        eval = sum(int(x == y) for (x, y) in test_results)
        print(test_results)
        print("{0} / {1}".format(eval, n_test))


#predict
#def predict:


print("**predict species of iris")
data = load_iris("Iris.csv")
np.random.shuffle(data)
training, testing = data[:135,:], data[135:,:]

training_inputs = training[:,0:len(training[0]) - 1]
training_outputs = training[:,len(training[0]) - 1:].flatten()
training_data = list(zip(training_inputs, training_outputs))

testing_inputs = testing[:,0:len(testing[0]) - 1]
testing_outputs = testing[:,len(testing[0]) - 1:].flatten()
testing_data = list(zip(testing_inputs, testing_outputs))

# print(training_data)
net = Network([4,4,3])
eta = 50
net.build_model(training_data, eta)
net.predict(testing_data)

#plot_decision_boundary()
