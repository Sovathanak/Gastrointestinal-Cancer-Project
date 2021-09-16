import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn import preprocessing
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

# Import extratcted features
IV3_features = pd.read_csv("extractedFeatures\\InceptionV3features.csv")
Res_features = pd.read_csv("extractedFeatures\\ResNet18features.csv")
VGG_features = pd.read_csv("extractedFeatures\\VGG16features.csv")

# Partition data into training and testing sets
# For now only using IV3 - can expland later
IV3_features = IV3_features.to_numpy()

X, y = IV3_features[:, 0:9], IV3_features[:, 10]

# Preprocessing test
X = preprocessing.scale(X)

X_train_IV3, X_test_IV3, y_train_IV3, y_test_IV3 = train_test_split(
    X, y, test_size=0.1, train_size=0.9
)

X_train_IV3, X_valid_IV3, y_train_IV3, y_valid_IV3 = train_test_split(
    X_train_IV3, y_train_IV3, test_size=float(1 / 9), train_size=float(8 / 9)
)
print("Data split done.")
print(X_train_IV3.shape)


def nnet(input_dims, output_dims, act_func):
    dnn_model = Sequential()
    dnn_model.add(Dense(units=20, input_shape=(input_dims,), activation=act_func))
    dnn_model.add(Dense(units=25, activation=act_func))
    dnn_model.add(Dense(units=25, activation=act_func))
    dnn_model.add(Dense(units=output_dims, activation="softmax"))
    return dnn_model


nnet = nnet(9, 1, "relu")
nnet.build()
nnet.summary()

nnet.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
nnet.fit(
    x=X_train_IV3,
    y=y_train_IV3,
    batch_size=32,
    epochs=20,
    validation_data=(X_valid_IV3, y_valid_IV3),
)
nnet.evaluate(X_test_IV3, y_test_IV3)
