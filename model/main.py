import os
import h5py
import numpy as np
from keras.models import load_model
from buildCNN import *

def getModel(path, filename):
    try:
        return (load_model("{}\\{}".format(path, filename.format("_trained.h5"))), 0)
    except:
        print("Trained model does not exists, attempt to open untrained model")
        try:
            return (load_model("{}\\{}".format(path, filename.format(".h5"))), 1)
        except:
            print("Untrained model does not exist, build new model")
            return (None, 2)


if __name__=="__main__":

    PATH = "C:\\Users\\Yash\\Desktop\\ece57000\\Project\\model\\generated_models"
    filename = "model_1{}"
    model, typeModel = getModel(PATH, filename.format)
    if not model:
        dimensions = (1, 224, 224)
        model = buildModel(dimensions, filename, scaling=2.606) # Holdover from Colorful Image Colorization
        model.save("{}\\{}".format(PATH, filename.format(".h5")))
    elif typeModel == 1:
        # Train network
        pass
    else:
        pass
    