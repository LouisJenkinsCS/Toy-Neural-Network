# A single-layer perceptron built for hand-recognition
# NOTE: Currently is not working appropriately, requires more work...
# Author's note: I think this isn't an appropriate problem for a perceptron...

import numpy as num
import scipy.special
import reader

iterations = 10


# Activation function
def sigmoid(x):
    return 1 / (1 + scipy.special.expit(-x))


# Slope for activation function
def slope(x):
    return x * (1-x)


# The domain is our list of inputs; each input consists of 3 integers
domain = reader.readImages()

# The codomain is our list of outputs; each output is index-mapped
# to its input, so domain[i] has output codomain[i]
codomain = []
for label in reader.readLabels():
    arr = num.zeros(9)
    arr[label - 1] = 1
    codomain.append(arr)
codomain = num.array(codomain)

# Deterministic Seed
num.random.seed(1)

# Our weight matrix that represents weights between edges in our graph
# The weights determine how likely the synapses are going to fire,
# and so the heavier the weight the more likely; this will be changed
# during back propagation corrections.
weights = 2 * num.random.random((domain.shape[-1], 9)) - 1

for i in range(iterations):
    # forward propagation
    layer0 = domain
    # Perform dot product to adjust the input so that the sigmoid function will
    # produce variable results. I.E If the weight of neuron 'n' is 0, then
    # sigmoid(0) = 0.5; the weight will adjust so that the sigmoid value will
    # become appropriate.
    layer1 = sigmoid(num.dot(layer0, weights))

    # Find error
    layerError1 = codomain - layer1

    # We adjust based on how correct and how 'sure' the machine was;
    # if the machine was sure (~1) but was incorrect, it will adjust
    # a lot more than if it was unsure (~0). If it was correct, then
    # there would be no change (error would be 0).
    layerDelta1 = layerError1 * slope(layer1)

    # Update the synaptic weights
    weights += num.dot(layer0.T, layerDelta1)

    # Debug Print
    if (i == 0):
        print("[Initial] Layer Error (Avg): {0:.0f}%"
              .format(num.average(layerError1) * 100))
    elif (i == iterations-1):
        print("[Final] Layer Error (Avg): {0:.0f}%"
              .format(num.average(layerError1) * 100))
        print("Weights: ", weights)
