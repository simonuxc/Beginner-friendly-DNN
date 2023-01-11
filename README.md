# Introductory to DNN
A neural network written in Python to solve XOR problem.

# Architecture
Dimensions of this neural network can be changed dynamically. For XOR problem it is sufficcient to have 2 neurons in the input layer, 10 neurons in the hidden layer and 2 neurons in the output layer (classes '0' and '1'). Dimensions are adjusted by this line of code:
```
NN_dimensions = [2, 10, 10, 2]
```
In this example number of iterations is fixed, the learning rate is fixed as well. My example should display the essence of neural networks with hidden layers, that is math.
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
\hat{y_i} = a_i^{(lastLayer)} 
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
## Sum, activation and cost functions for forward propoagation
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
C = \frac{1}{2}*\sum_{k=1}^M(y_i - \hat{y_i})^2
```
## Derivatives of cost function - calculating gradient
Now onto derivatives. In order to calculate gradient for each weight we have to calculate the partial derivative of cost with respect to that weight. Therefore, we want to find:
```math
\frac{\partial{C}}{\partial{W_{ij}}^{(n)}}
```
If we are calculating for the last layer _n_, by following chain, previous derivative can be expanded to the following equation:
```math
\frac{\partial{C}}{\partial{W_{ij}^{(n)}}} = \frac{\partial{C}}{\partial{\hat{y_i}^{(n)}}} * \frac{\partial{\hat{y_i}^{(n)}}}{\partial{{z_{i}^{(n)}}}}  * \frac{\partial{z_{i}^{(n)}}}{\partial{{W_{(ij)}^{(n)}}}}
```
In order to do that, let's first find each derivative:
```math
\frac{\partial{C}}{\partial{\hat{y_i}^{(n)}}} = y_i - \hat{y_i}
```
```math
\frac{\partial{\hat{y_i}^{(n)}}}{\partial{{z_{i}^{(n)}}}} = \hat{y_i} * (1 - \hat{y_i})
```
```math
\frac{\partial{z_{i}^{(n)}}}{\partial{{W_{(ij)}^{(n)}}}} = a_j^{(n-1)}
```
Putting it all back together, we get the full gradient:
```math
grad = \frac{\partial{C}}{\partial{W_{ij}^{(n)}}} = (y_i - \hat{y_i}) * (\hat{y_i} * (1 - \hat{y_i})) * a_j^{(n-1)}
```
If we want to adjust the bias, then the partial derivative of sum with respect to bias is 1; we get:
```math
grad = \frac{\partial{C}}{\partial{b_{i}^{(n)}}} = (y_i - \hat{y_i}) * (\hat{y_i} * (1 - \hat{y_i}))
```
That's it! Having found the gradient for each weight in the ouptut layer, we can use it to adjust the weights:
```math
W_{ij}^{(n)} := W_{ij}^{(n)} - grad * learningRate
```
Now the harder part is calculating the gradient for hidden layers. Luckily, there is a pattern: derivatives repeat themselves, hence calculating and saving derivatives as we move from the outer layer (BACK propagation) to 'front' enables us to reuse the values (NOTE: we take the derivative with respect to sum, not with respect to weight. We do this so we can save the partial derivative for calculating derivatives at different level. _M_ - number of neurons in the second (n+1) layer).
```math
\frac{\partial{C}}{\partial{{z_i^{(n)}}}} = \sum_{k=1}^{M}\frac{\partial{C}}{\partial{a_k^{(n+1)}}} * \frac{\partial{a_k^{(n + 1)}}}{\partial{z_k^{(n+1)}}} * \frac{\partial{z_k^{(n+1)}}}{\partial{{a_i^{(n)}}}} * \frac{\partial{a_i^{(n)}}}{\partial{{z_{i}^{(n)}}}} 
```
Note how the last derivative is constant in terms of _k_ (that value is the same for every multipliciation in sum function), so we can put it before sum notation. Also note, that we have two partial derivatives that can be rewriten into one:
```math
\frac{\partial{C}}{\partial{a_k^{(n+1)}}} * \frac{\partial{a_k^{(n + 1)}}}{\partial{z_k^{(n+1)}}} = \frac{\partial{C}}{\partial{z_k^{(n+1)}}}
```
We get:
```math
\frac{\partial{C}}{\partial{{z_i^{(n)}}}} = \frac{\partial{a_i^{(n)}}}{\partial{{z_{i}^{(n)}}}} * \sum_{k=1}^{M}\frac{\partial{C}}{\partial{z_k^{(n+1)}}} * \frac{\partial{z_k^{(n+1)}}}{\partial{{a_i^{(n)}}}}
```
```math
\frac{\partial{a_i^{(n)}}}{\partial{{z_{i}^{(n)}}}} = a_i^{(n)} * (1 - a_i^{(n)})
```
```math
\frac{\partial{z_k^{(n+1)}}}{\partial{{a_i^{(n)}}}} = W_{ji}^{(n+1)}
```
```math
\frac{\partial{C}}{\partial{{z_i^{(n)}}}} = a_i^{(n)} * (1 - a_i^{(n)}) * \sum_{k=1}^{M}\frac{\partial{C}}{\partial{z_k^{(n+1)}}} * W_{ji}^{(n+1)}
```
Notice how for the cost derivative with respect to sum in layer n we have to use cost derivative with respect to sum in layer (n+1). That's the tricky part - saving the derivatives. If we go back to derivative in the last (output) layer we see, that we have calculated full gradient for each weight. But from the general formula we also see, that only cost derivative with respect to sum is needed.
```math
\frac{\partial{C}}{\partial{{z_{i}^{(n)}}}} = \frac{\partial{C}}{\partial{\hat{y_i}^{(n)}}}  * \frac{\partial{\hat{y_i}^{(n)}}}{\partial{{z_{i}^{(n)}}}}
```
```math
\frac{\partial{C}}{\partial{{z_{i}^{(n)}}}} = (y_i - \hat{y_i}) * \hat{y_i} * (1 - \hat{y_i})
```
Now we have everything we need for our gradient descent algorithm. In the code, we have to save these derivatives in an array. Once they are calculated, weights can be adjusted by this final formula (note, that derivative for the output layer has to be calculated separately, but all others follow the same convention):
```math
\frac{\partial{C}}{\partial{W_{ij}^{(n)}}} = \frac{\partial{C}}{\partial{{z_{i}^{(n)}}}} * a_j^{(n-1)}
```
