# Beginner-friendly-DNN
A neural network written in Python to solve XOR problem.

# Architecture
Dimensions of this neural network can be changed dynamically. For XOR problem it is sufficcient to have 2 neurons in the input layer, 10 neurons in the hidden layer and 2 neurons in the output layer (classes '0' and '1'). Dimensions are changed here:

# Calculations
Notations used in formulas:

Weight notation, where (n) - layer number, i - neuron number, j - weight number:
```math
W_{ij}^{(n)}
```
The sum value z of a neuron, where (n) - layer number, i - neuron number:
```math
z_{i}^{(n)}
```
The activation value a of a neuron, where (n) - layer number, i - neuron number:
```math
a_{i}^{(n)}
```
The sum z for each neuron is calculated by sum formula.
If we wish to calculate z<sub>ij</sub><sup>(n)</sup>, we use the sum formula:
```math
z_{ij}^{(n)} = b_{i}^{(n)} + \sum_{k=1}^{M}(W_{ik}^{(n)}*X_k^{(n-1)})
```
Activation function of every neuron in the neural network is sigmoid, where a<sub>i</sub> is that activation value and z<sub>i</sub> is the sum value of i neuron
**Sigmoid**
```math
a_i = 1 / (1 + e^z_i)
```

The cost function used here is slightly different from MSE (Mean Square Error):

