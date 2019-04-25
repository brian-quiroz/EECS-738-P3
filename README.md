# EECS-738-P3
## how neural network can be used to make classification prediction

### architecture

In this project, we utilize a simple neural network (more precisely, a multilayer perceptron, or MLP) to perform two classification tasks. This neural network consists of one input layer, one hidden layer and one output layer. Our datasets were the "Iris Dataset" and the "Seeds Dataset".

The input layer's dimension is equal to the dimension of the input data (amount of features we consider when classifying an item). The output layer's dimension is equal to the dimension fo the output data (amount of classes we classify our input into)

There is no best solution to determine the dimension of the hidden layer, which is usually determined empirically. According to ***Kolmogorov theorem***, we set it to the 2n + 1, where n is the dimension of input layer. The activation function we choose for hidden layer is ***tanh***.

To find best parameters for training model, difference between prediction and expected class needs to be minimized. We use ***cross-entropy loss*** to define this difference. To minimize the total loss, we employ ***batch gradient descent***. The back propagation algorithm will be used to calculate gradient efficiently.

We tried different values for learning rate and regularization strength. It turns out that when learning rate == 0.001 and regularization strength == 0.001, comparable decent results can be achieved.

Finally, to make sure the final probabilities is in range [0, 1], we apply activation function *softmax* on output layer.

##Iris Dataset
The dimension of the input layer for the Iris Dataset was 4, since each iris item consists of four features: SepalLength, SepalWidth, PetalLength and PetalWidth. The output layer would be three dimensional since we are classifying flowers into 3 categories. To make it computational feasible,  we replace "Iris-setosa" with 0, 'Iris-versicolor' with 1, and 'Iris-virginica' with 2.


##Seeds Dataset
The dimension of the input layer for the Seeds Dataset was 7, since each seed item consists of four features: Area, Perimeter, Compactness, Length of Kernel, Width of Kernel, Asymmetry Coefficient, Length of Kernel Groove. The output layer would be three dimensional since we are classifying seeds into three types of wheat (represented in the dataset as integers from 1 to 3). For consistency with our implementation, we subtracted 1 from each output integer such that 3 was mapped to 2, 2 mapped to 1, and 1 mapped to 0.


### References

http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
http://neuralnetworksanddeeplearning.com/chap1.html
