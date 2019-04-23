# EECS-738-P3
how neural network can be used to make classification prediction

###architecture
In this project, we utilize neural network to generate classification prediction for input data. This neural network will be consisted of one input layer, one hidden layer and one output layer. In following discussion, input dataset of iris is predetermined.

The dimension of input layer, namely the number of input nodes in input layer, is determined by the dimension of data. Hence, dimension of input layer would be 4, since each item consists four features of SepalLength, SepalWidth, PetalLength and PetalWidth.

The dimension of output layer is determined by the number of classes. So output layer would be three dimensional. To make it computational feasible,  we assumes replace "Iris-setosa" with 0, 'Iris-versicolor' with 1, and 'Iris-virginica' with 2.

The dimension of hidden layer are determined empirically. According to Kolmogorov theorem, we set it to the 2*n + 1, where n is the dimension of input layer. 

we tried different values for learning rate and regularization strength. It turns out that when learning rate == 0.001 and regularization strength == 0.001, comparable decent results can be achieved.

To make sure the final probabilities is in range [0, 1], we apply activation function softmax on output layer.



###reference
http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
http://neuralnetworksanddeeplearning.com/chap1.html