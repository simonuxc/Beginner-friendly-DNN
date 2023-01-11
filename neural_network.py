import random
import numpy as np
import matplotlib.pyplot as plt

# activation function of one neuron
def sigmoid(value):
    return 1 / (1 + np.exp(value))

# sum calculation for one neuron
def sum_z(x, weights):
    sum = 0
    for i in range(len(x)):
        sum += x[i] * weights[i]
    return sum

def calculate_activations(neurons_activations, x_set, weights):
    # set acitvations of the '0' layer as input values
    for i in range(len(neurons_activations[0])):
        neurons_activations[0][i] = x_set[i]
    # calculate activations of all neurons, skip the 'input layer'
    # i is the current layer
    for i in range(1, len(neurons_activations)):
        # j is the number of neurons in the previous layer
        for j in range(len(neurons_activations[i])):
            x = [1] + neurons_activations[i - 1] # add one in the beggining for the bias
            # to the sum function we pass inputs from previous layer and weights of current layer
            neurons_activations[i][j] = sum_z(x, weights[i - 1][j])
            neurons_activations[i][j] = sigmoid(neurons_activations[i][j])
    # neural netwok predictions
    y_hat = neurons_activations[-1]
    return y_hat

# return square error based on current output
def se(y_hat, y_train):
    return (y_train[0] - y_hat[0])**2 + (y_train[1] - y_hat[1])**2

# training data
x_train = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_train = [[1, 0], [0, 1], [0, 1], [1, 0]]

NN_dimensions = [2, 10, 2]
iterations = 10000
step = 0.1
# prepare square errors array for each training scenario
se_arr = []
for i in range(len(x_train)): se_arr.append([])

# create an array for keeping activation values. The activations of first layer will be initial inputs
neurons_activations = []
for i in range(len(NN_dimensions)):
    neurons_activations.append([])
    for j in range(NN_dimensions[i]):
        neurons_activations[i].append(0)

# create an array for keeping neuron weights. The '0' (input) layer doesn't have weights. Weights[i][j][k]: i - layer, j - neuron, k - weight
neurons_weights = []
# i is the number of layer
for i in range(len(NN_dimensions) - 1):
    neurons_weights.append([])
    # j is the amount of neurons each layer has
    for j in range(NN_dimensions[i + 1]):
        neurons_weights[i].append([])
        # k is the number of weight (the number of neurons in the previous layer) + 1 bias
        for k in range(NN_dimensions[i] + 1):
            neurons_weights[i][j].append(random.random() - 0.5)

# create an array for keeping derivatives of neurons (dcdz, where z is the sum of current layer)
derivatives_dcdz = []
# the first layer doesn't need derivative to be calculated, because it has no weights
for i in range(len(NN_dimensions) - 1):
    derivatives_dcdz.append([])
    for j in range(NN_dimensions[i + 1]):
        derivatives_dcdz[i].append(0)

# train the model
for itr in range(iterations):
    # track training progress
    if itr % (iterations / 20) == 0:
        print("Training at: {:.2f}%".format((itr / iterations * 100)))
    
    # go through each training scenario
    for train_id in range(len(x_train)):
        # forward prop is done, all activations values are calculated
        y_hat = calculate_activations(neurons_activations, x_train[train_id], neurons_weights)
        # add square error to the se_array
        se_arr[train_id].append(se(y_hat, y_train[train_id]))

        # reset derivatives array
        for i in range(len(derivatives_dcdz)):
            for j in range(len(derivatives_dcdz[i])):
                derivatives_dcdz[i][j] = 0

        # calculate derivatives of cost with respect to every sum of the last (output) layer
        for i in range(len(derivatives_dcdz[-1])):
            derivatives_dcdz[-1][i] = ((y_train[train_id][i] - y_hat[i]) * (1 - y_hat[i]) * y_hat[i])

        # calculate derivatives of all the other hidden layers
        # go through each hidden layer starting at the one before last, i is the index of that layer
        for i in range(len(derivatives_dcdz) - 2, -1, - 1):
            # go through each of its 'neuron' derivative
            for j in range(len(derivatives_dcdz[i])):
                # calculate part of that derivative using derivative formula (sum of derivatives in the following layer * respective weights)
                for k in range(len(derivatives_dcdz[i + 1])):
                    # I believe it should be ... = ... * neurons_weights[i + 1][k][j + 1], because neurons_weights[i + 1][k][0] is bias value
                    # for which we don't have to account in derivative, yet writing 'j + 1' gives shit answers
                    derivatives_dcdz[i][j] += derivatives_dcdz[i + 1][k] * neurons_weights[i + 1][k][j]
            # go through each derivative in layer i and 'complete' it (formula)
            for j in range(len(derivatives_dcdz[i])):
                # neurons_activations[i + 1] - activations of this layer
                derivatives_dcdz[i][j] *= (neurons_activations[i + 1][j]) * (1 - neurons_activations[i + 1][j])

        # using derivatives adjust weights
        # go through each layer i
        for i in range(len(neurons_weights)):
            # go through each neuron j
            for j in range(len(neurons_weights[i])):
                # go through each weight k
                for k in range(len(neurons_weights[i][j])):
                    # adjust bias
                    if k == 0:
                        # in the sum formula bias is multiplied by '1', so the full gradient is just dcdz
                        grad = derivatives_dcdz[i][j] * 1
                        neurons_weights[i][j][k] -= grad * step
                    # adjust all other weights
                    else:
                        # in the sum formula wieght is multiplied by the ouput of prievious layer
                        grad = derivatives_dcdz[i][j] * neurons_activations[i][k - 1]
                        neurons_weights[i][j][k] -= grad * step

# Test the network on same inputs
for i in range(len(x_train)):
    y_hat = calculate_activations(neurons_activations, x_train[i], neurons_weights)
    print("Test data:        ", x_train[i])
    print("Expected results: ", y_train[i])
    print("Predicted results:", y_hat)
    print()
    plt.plot(se_arr[i])
    plt.show()