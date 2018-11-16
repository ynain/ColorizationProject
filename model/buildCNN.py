import numpy
import keras
from keras import Model
from keras.layers import Input, Dense, Activation, Dropout, Flatten, LSTM
from keras.layers import MaxPool2D, Conv2D, Conv2DTranspose, merge, BatchNormalization, Softmax

def buildModel(dimensions, cnnName, scaling):
    model = buildCNN(dimensions, cnnName, scaling)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def buildCNN(dimensions, cnnName, scaling):
    # Initial Layer
    inputdata = Input(shape=dimensions, name="input_data")

    # Hidden Blocks
    # Input format - model, input layer, filters, [(kernel_size, strides, padding, dilation_rate)], blockID
    model = buildCNNBlock(model, inputdata, 64, [(3, 1, 1, 1), (3, 2, 1, 1)], [1, 1])
    model = buildCNNBlock(model, model, 128, [(3, 1, 1, 1), (3, 2, 1, 1)], [2, 1])

    model = buildCNNBlock(model, model, 256, [(3, 1, 1, 1), (3, 1, 1, 1), (3, 2, 1, 1)], [3, 1])
    model = buildCNNBlock(model, model, 512, [(3, 1, 1, 1), (3, 1, 1, 1), (3, 1, 1, 1)], [4, 1])

    # 5 and 6 are for dilation
    model = buildCNNBlock(model, model, 512, [(3, 1, 2, 2), (3, 1, 2, 2), (3, 1, 2, 2)], [5, 1])
    model = buildCNNBlock(model, model, 512, [(3, 1, 2, 2), (3, 1, 2, 2), (3, 1, 2, 2)], [6, 1])

    model = buildCNNBlock(model, model, 256, [(3, 1, 1, 1), (3, 1, 1, 1), (3, 1, 1, 1)], [7, 1])
    # 8.1 is deconvolution
    model = Conv2DTranspose(256, kernel_size=4, strides=2, activation="relu", name="conv8_1_deconv")(model)
    model = buildCNNBlock(model, model, 128, [(3, 1, 1, 1), (3, 1, 1, 1)], [8, 2])
    
    # Softmax block
    model = Conv2D(313, kernel_size=1, strides=1, padding=1, dilation_rate=1, name="ab_313")(model)
    model = BatchNormalization(scale=scaling)
    model = Softmax(name="softmax")(model)

    # Decode output
    model = Conv2D(2, kernel_size=1, strides=1, dilation_rate=1, name="ab_2")(model)

    network = Model(input=[inputdata], output=[model], name=cnnName)
    
    return network

def buildCNNBlock(m, inputLayer, numFilters, parameters, blockID):
    for ks, s, p, d in parameters:
        m = Conv2D(numFilters, kernel_size=ks, strides=s, padding=p, dilation_rate=d, activation="relu", name="conv{}_{}".format(blockID))(inputLayer)
        inputLayer = m
        blockID[1] += 1
    if (blockID[0] != 8):
        m = BatchNormalization(name="batchnorm_{}".format(blockID[0]))(m)
    return m