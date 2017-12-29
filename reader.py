import io
import numpy

# Reads in the dataset provided by MNIST as a matrix of 28x28 pictures.
# Labels are separate from the images themselves and so are contained in a
# separate array where for each index 'i', label[i] is the label for image[i]

imageFile = "train-images.idx3-ubyte"
labelFile = "train-labels.idx1-ubyte"


def readInt(handle):
    return int.from_bytes(handle.read(4), byteorder='big')


def readImages():
    with open(imageFile, 'rb') as f:
        # Discard Magic
        readInt(f)

        # Metadata
        nImages = readInt(f)
        nRows = readInt(f)
        nCols = readInt(f)

        # Read data
        arr = numpy.reshape(bytearray(f.read()), (nCols, nRows, nImages))
        arr = numpy.moveaxis(arr, [0, 1, 2], [1, 0, 2])
        arr = numpy.reshape(arr, (arr.shape[0], arr.shape[1], arr.shape[2]))
        arr = arr / float(255)
        arr = arr.reshape(-1, arr.shape[-1])
        arr = arr.reshape((arr.shape[1], arr.shape[0]))
        return arr


def readLabels():
    with open(labelFile, 'rb') as f:
        # Discard Magic
        readInt(f)

        # Metadata
        nLabels = readInt(f)

        # Read data
        return numpy.array([int(x) for x in bytearray(f.read())])
