# Beginner-friendly-DNN
A neural network written in Python to solve XOR problem.

# Architecture
Dimensions of this neural network can be changed dynamically. For XOR problem it is sufficcient to have 2 neurons in the input layer, 10 neurons in the hidden layer and 2 neurons in the output layer (classes '0' and '1'). Dimensions are changed here:

# Notations used in formulas for gradient calculation:

Weight notation, where _(n)_ - layer index, _i_ - neuron index, _j_ - weight index:
```math
W_{ij}^{(n)}
```
Bias value, where _(n)_ - layer index, _i_ - neuron index:
```math
b_{i}^{(n)}
```
The sum value _z_ of a neuron, where _(n)_ - layer index, _i_ - neuron index:
```math
z_{i}^{(n)}
```
The activation value _a_ of a neuron, where _(n)_ - layer index, _i_ - neuron index:
```math
a_{i}^{(n)}
```
The predicted output for class _i_, where _i_ is the index of neuron in the last (output) layer:
```math
\hat{y_i} = a_i^{(last_layer)} 
```
The expected output for class _i_, where _i_ is the index of neuron in the last (output) layer:
```math
y_i
```
The cost function of whole neural network:
```math
C
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

The cost function used here is slightly different from MSE (Mean Square Error). Instead of taking the mean of all square errors, we take 1/2. This is done in order for the derivative to be a bit simplier.
MSE (or _C_)is calculated as follows:
```math
C = \frac{1}{2}*\sum_{k=1}^M(\hat{y_i} - y_i)^2
```
Now onto derivatives. In order to calculate gradient for each weight we have to calculate the partial derivative of cost with respect to that weight. In order to do that, let's first derive general expressions for:
1. Partial derivative of cost function with respect to _z<sub>i</sub>_ in the last layer
2. Partial derivative of _a<sub>i</sub><sup>(n)</sup>_ with the respect to _z<sub>i</sub><sup>(n)</sup>_
3. Partial derivative of _z<sub>i</sub><sup>(n)</sup>_ with respect to _w<sub>ij</sub><sup>(n)</sup>_
## Derivatiion
\frac{\partial{C}}{/partial{z_i}}
