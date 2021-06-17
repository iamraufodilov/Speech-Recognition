# load libraries
from preprocessing import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from keras.utils import to_categorical
import tensorflow as tf
#import wandb
#from wandb.keras import WandbCallback
import matplotlib.pyplot as plt
#import config


# load data

max_len = 11
buckets = 20

# Save data to array file first
save_data_to_array(max_len=max_len, n_mfcc=buckets)
labels=["bird", "cat", "dog"]

# Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()

# Feature dimension
channels = 1
epochs = 10
batch_size = 10

num_classes = 3

X_train = X_train.reshape(X_train.shape[0], buckets, max_len, channels)
X_test = X_test.reshape(X_test.shape[0], buckets, max_len, channels)


y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)


# create the model
model = Sequential()
model.add(Flatten(input_shape=(buckets, max_len)))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])


# get callbacks
class MyCallback(tf.keras.callbacks.Callback):
  def on_train_end(self, logs=None):
    global training_finished
    training_finished = True


# train the model
model.fit(X_train, y_train_hot, epochs=epochs, validation_data=(X_test, y_test_hot), callbacks=[MyCallback()])

#here we go our model's accuracy is not hihg cuz we use only 60 audio file which too less for model
# anyway goood job behind of cencept speech recognition