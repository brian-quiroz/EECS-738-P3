# EECS-738-P3
## how neural network can be used to make classification prediction

### architecture

In this project, we utilize a simple neural network to generate classification prediction. This neural network consists of one input layer, one hidden layer and one output layer. In following discussion, input dataset of iris is predetermined.

The dimension of input layer, namely the number of input nodes in input layer, is determined by the dimension of data. Thus, dimension of input layer would be 4, since each iris item consists four features of SepalLength, SepalWidth, PetalLength and PetalWidth. The dimension of output layer is determined by the number of classes. So in this case output layer would be three dimensional. To make it computational feasible,  we replace "Iris-setosa" with 0, 'Iris-versicolor' with 1, and 'Iris-virginica' with 2. There is no best solution to determine the dimension of hidden layer, which is usually determined empirically. According to ***Kolmogorov theorem***, we set it to the 2n + 1, where n is the dimension of input layer. The activation function we choose for hidden layer is ***tanh***

To find best parameters for training model, difference between prediction and expected class needs to be minimized. We use ***cross-entropy loss*** to define this difference. To minimize total loss, we employ ***batch gradient descent***. Back propagation algorithm will be used to calculate gradient efficiently.

we tried different values for learning rate and regularization strength. It turns out that when learning rate == 0.001 and regularization strength == 0.001, comparable decent results can be achieved.

Finally, to make sure the final probabilities is in range [0, 1], we apply activation function *softmax* on output layer.



### reference

http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
http://neuralnetworksanddeeplearning.com/chap1.html
