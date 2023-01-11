# Beginner-friendly-DNN
A neural network written in Python to solve XOR problem.

# Architecture
Dimensions of this neural network can be changed dynamically. For XOR problem it is sufficcient to have 2 neurons in the input layer, 10 neurons in the hidden layer and 2 neurons in the output layer (classes '0' and '1'). Dimensions are changed here:

# Notations used in formulas for gradient calculation:

Weight notation, where _(n)_ - layer number, _i_ - neuron number, _j_ - weight number:
```math
W_{ij}^{(n)}
```
Bias value, where _(n)_ - layer number, _i_ - neuron number:
```math
b_{i}^{(n)}
```
The sum value _z_ of a neuron, where _(n)_ - layer number, _i_ - neuron number:
```math
z_{i}^{(n)}
```
The activation value _a_ of a neuron, where _(n)_ - layer number, _i_ - neuron number:
```math
a_{i}^{(n)}
```
# Calculations
It is necessary to calculate the sum values _z_ of every neuron in every hidden layer.
We use the formula as follows (_M_ - number of neurons in the previous layer):
```math
z_{i}^{(n)} = b_{i}^{(n)} + \sum_{k=1}^{M}(W_{ik}^{(n)}*a_k^{(n-1)})
```
Activation function of every neuron in this neural network is sigmoid.
It is calculated as follows:
```math
a_i = \frac{1}{1 + e^{(-z_i)}}
```

The cost function used here is slightly different from MSE (Mean Square Error):

