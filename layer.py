import numpy as num

iterations = 10000


# Activation function
def sigmoid(x):
    return 1 / (1 + num.exp(-x))


# Slope for activation function
def slope(x):
    return x * (1-x)


# The domain is our list of inputs; each input consists of 3 integers
domain = num.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])
# The codomain is our list of outputs; each output is index-mapped
# to its input, so domain[i] has output codomain[i]
codomain = num.array([
    [0], [0], [1], [1]
])

# Deterministic Seed
num.random.seed(1)

# Our weight matrix that represents weights between edges in our graph
# The weights determine how likely the synapses are going to fire,
# and so the heavier the weight the more likely; this will be changed
# during back propagation corrections.
weights = 2 * num.random.random((3, 1)) - 1
print("Expected Answer: ", codomain)

for i in range(iterations):
    # forward propagation
    layer0 = domain
    # Perform dot product to adjust the input so that the sigmoid function will
    # produce variable results. I.E If the weight of neuron 'n' is 0, then
    # sigmoid(0) = 0.5; the weight will adjust so that the sigmoid value will
    # become appropriate.
    layer1 = sigmoid(num.dot(layer0, weights))
    if (i == 0):
        print("Initial Guess: ", layer1)
    elif (i == iterations-1):
        print("Final Guess: ", layer1)

    # Find error
    layerError1 = codomain - layer1

    # We adjust based on how correct and how 'sure' the machine was;
    # if the machine was sure (~1) but was incorrect, it will adjust
    # a lot more than if it was unsure (~0). If it was correct, then
    # there would be no change (error would be 0).
    layerDelta1 = layerError1 * slope(layer1)

    # Update the synaptic weights
    weights += num.dot(domain.T, layerDelta1)

    # print(weights)

# forward propagation
layer0 = num.array([[1, 1, 1]])
layer1 = sigmoid(num.dot(layer0, weights))
# Find error
layerError1 = num.array([[1]]) - layer1

print(layerError1)
