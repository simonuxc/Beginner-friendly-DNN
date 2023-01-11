# Beginner-friendly-DNN
A neural network written in Python to solve XOR problem.

# Architecture
Dimensions of this neural network can be changed dynamically. For XOR problem it is sufficcient to have 2 neurons in the input layer, 10 neurons in the hidden layer and 2 neurons in the output layer (classes '0' and '1'). Dimensions are changed here:

# Calculations
Activation function of every neuron in the neural network is sigmoid, where a<sub>i</sub> is the activation value and z<sub>i</sub> is the sum value of i neuron
**Sigmoid**
$$\A_i(Z_i) = 1 / (1 + e^-Z_i)

The cost function used here is slightly different from MSE (Mean Square Error):

