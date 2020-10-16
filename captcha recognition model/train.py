import keras
from keras.models import load_model

print(keras.__version__)
path = './model/cnn_model.hdf5'
model = load_model(path)
model.summary()
