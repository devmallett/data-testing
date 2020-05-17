import tensorflow as tf
from tensorflow import keras
# from tensorflow import ConfigProto, Session
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import keras.datasets
from keras import backend as K

# python ten-cnn.py
# https://www.youtube.com/watch?v=WvoLTXIjBYU

#Running Many GPUs at the same time
# gpu_options = 
# gpu_options = tf.compat.v1.ConfigProto.GPUoptions(per_process_gpu_memory_fraction=0.333)
# # config = tf.compat.v1.ConfigProto(gpu_options = (per_process_gpu_memory_fraction=0.3))
# sess = tf.compat.v1.ConfigProto.Session(config=gpu_options)

X = pickle.load(open("X.pickle", "rb")) #images
y = pickle.load(open("y.pickle", "rb")) #labels

# print(len(X))
# print(len(y))



X = X/255.0 

def swish_activation(x):
    return (K.sigmoid(x)*x)


# get_custom_objects().update({'swish_activation': Activation(swish_activation)})

#accuracy 0.7343 | 0.7375 | 0.7355
# model = keras.Sequential([
#     keras.layers.Conv2D((64), (3,3), input_shape = X.shape[1:]),
#     keras.layers.Activation("selu"), #accuracy #.7343 
#     # tf.keras.activations.swish_activation(X),
#     keras.layers.MaxPooling2D(pool_size=(2,2)),
#     # ------------------------------- end of layer 1
#     keras.layers.Conv2D((64), (3,3)),
#     keras.layers.Activation("selu"),
#     # tf.keras.activations.swish_activation(X),
#     keras.layers.MaxPooling2D(pool_size=(2,2)), #2 x 2 filter, derivative, used to reduce 
#     # --------------------------------- end of layer 2
#     keras.layers.Flatten(),
#     keras.layers.Dense(64),
#     keras.layers.Activation("selu"),
#     # tf.keras.activations.swish_activation(X),
#     keras.layers.Dense(1),
#     keras.layers.Activation("sigmoid"),

#     # when does one use a dense layer?
# ])

#accuracy 
model = keras.Sequential([
    keras.layers.Conv2D((64), (3,3), input_shape = X.shape[1:]),
    keras.layers.Activation("relu"), #accuracy #.7343 
    # tf.keras.activations.swish_activation(X),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    # ------------------------------- end of layer 1
    keras.layers.Conv2D((64), (3,3)),
    keras.layers.Activation("relu"),
    # tf.keras.activations.swish_activation(X),
    keras.layers.MaxPooling2D(pool_size=(2,2)), #2 x 2 filter, derivative, used to reduce 
    # --------------------------------- end of layer 2
    keras.layers.Flatten(),
    keras.layers.Dense(64),
    keras.layers.Activation("relu"),
    # tf.keras.activations.swish_activation(X),
    keras.layers.Dense(1),
    keras.layers.Activation("sigmoid"),

    # when does one use a dense layer?
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X, y, batch_size=32, epochs = 3, validation_split=0.1)

# python ten-cnn.py


