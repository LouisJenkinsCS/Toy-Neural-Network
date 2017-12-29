# A simple multi-layer perceptron

import numpy as num
import scipy.special
import reader

iterations = 5
hiddenNeurons = 100


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
weights = 2 * num.random.random((domain.shape[-1], hiddenNeurons)) - 1
hiddenWeights = 2 * num.random.random((hiddenNeurons, codomain.shape[-1])) - 1

for i in range(iterations):
    # forward propagation
    layer0 = domain
    # Perform dot product to adjust the input so that the sigmoid function will
    # produce variable results. I.E If the weight of neuron 'n' is 0, then
    # sigmoid(0) = 0.5; the weight will adjust so that the sigmoid value will
    # become appropriate.
    layer1 = sigmoid(num.dot(layer0, weights))
    layer2 = sigmoid(num.dot(layer1, hiddenWeights))

    # Find error
    layerError2 = codomain - layer2

    # We adjust based on how correct and how 'sure' the machine was;
    # if the machine was sure (~1) but was incorrect, it will adjust
    # a lot more than if it was unsure (~0). If it was correct, then
    # there would be no change (error would be 0).
    layerDelta2 = layerError2 * slope(layer2)

    layerError1 = num.dot(layerDelta2, hiddenWeights.T)
    layerDelta1 = layerError1 * slope(layer1)
    print(num.average(layerError1) * num.average(slope(layer1)))
    print("Delta1=", num.average(layerDelta1), ", Delta2=", num.average(layerDelta2))

    # Update the synaptic weights
    hiddenWeights += num.dot(layer1.T, layerDelta2)
    print(layer1.T[0][0] * layerDelta2[0][0] + layer1.T[0][1] * layerDelta2[1][0])
    print(num.dot(layer1.T, layerDelta2)[0][0])
    weights += num.dot(layer0.T, layerDelta1)

    # Debug Print
    if (i == 0):
        print("[Initial] Layer1 Error (Avg): {0:.0f}%"
              .format(num.average(layerError1) * 100))
        print("[Initial] Layer2 Error (Avg): {0:.0f}%"
              .format(num.average(layerError2) * 100))
    elif (i == iterations-1):
        print("[Final] Layer1 Error (Avg): {0:.0f}%"
              .format(num.average(layerError1) * 100))
        print("[Final] Layer2 Error (Avg): {0:.0f}%"
              .format(num.average(layerError2) * 100))
