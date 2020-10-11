"""
    CNN.py  - This example trains a CNN on MNIST (or CFAR10) data

    Copyright (C) 2020 Adrian Bevan, Queen Mary University of London

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import api

#
# Training configuration
#
ValidationSplit = 0.5
BatchSize       = 20
Nepochs         = 20
DropoutValue    = 0.6

#
# Load the data chose between CFAR10 and MNIST examples
#
#   UseMNIST = False      - configure and train	using CFAR10
#   UseMNIST = True       - configure and train using MNIST
#
UseMNIST = False 
x_train  = None
y_train  = None
x_test   = None
y_test   = None
if UseMNIST:
  x_train, y_train, x_test, y_test = api.getMNIST()
  x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
  x_test  = x_test.reshape((x_test.shape[0], 28, 28, 1))

  print("shape = ", x_train.shape )
  print("shape = ", x_test.shape )
else:
  x_train, y_train, x_test, y_test = api.getCFAR10()

print("shape = ", x_train.shape )
print("shape = ", x_test.shape )

#
# as CFAR10 and MNIST are different shapes, extract the
#
xpix = x_train.shape[1]
ypix = x_train.shape[2]
zpix = x_train.shape[3] # color channels


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(xpix, ypix, zpix)))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10))

print("--------------------------------------------------------------------------------------------------------------")
if UseMNIST:
    print("\033[1mWill train a convolutional neural network on the MNIST data\033[0m")
else:
    print("\033[1mWill train a convolutional neural network on the CFAR10 data\033[0m")

print("--------------------------------------------------------------------------------------------------------------\n\n")
print("Input data MNIST")
print("Dropout values       = ", DropoutValue)
print("Leaky relu parameter =  0.1")
print("ValidationSplit      = ", ValidationSplit)
print("BatchSize            = ", BatchSize)
print("Nepochs              = ", Nepochs, "\n")
print("N(train)             = ", len(x_train))
print("N(test)              = ", len(x_test))
model.summary()


# now specify the loss function - cross entropy
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# now we can train the model to make predictions.
#   Use the ADAM optimiser
#   Specify the metrics to report as accuracy
#   Specify the loss function (see above)
# the fit step specifies the number of training epochs
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
history  = model.fit(x_train, y_train, validation_split=ValidationSplit, batch_size=BatchSize, epochs=Nepochs)

# Print out the history keys expected are:
#    loss        The loss function evaluated at each epoch for the training set
#    acc         The accuracy evaluated at each epoch for the training set
#    val_loss    The loss evaluated at each epoch for the validation set
#    val_acc     The accuracy evaluated at each epoch for the validation set
# The val_* entries exist only if there is a validation_split specified
print("history keys = ", history.history.keys())

print("\033[1mDisplay the evolution of the accuracy as a function of the training epoch\033[0m")
print("  N(Epochs)        = ", Nepochs)
print("  accuracy (train) = ", history.history['accuracy'])
print("  accuracy (test)  = ", history.history['val_accuracy'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
#plt.show()
print("Plotting the output to fig/CNN_accuracy_vs_epochs.*")
plt.savefig("fig/CNN_accuracy_vs_epochs.pdf")
plt.savefig("fig/CNN_accuracy_vs_epochs.png")
plt.clf()

print("\033[1mDisplay the evolution of the loss as a function of the training epoch\033[0m")
print("  N(Epochs)        = ", Nepochs)
print("  loss (train)     = ", history.history['loss'])
print("  loss (test)      = ", history.history['val_loss'])

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper right')
#plt.show()
print("Plotting the output to fig/CNN_loss_vs_epochs.*")
plt.savefig("fig/CNN_loss_vs_epochs.pdf")
plt.savefig("fig/CNN_loss_vs_epochs.png")

# having finished training the model, use this to evaluate the performance on a sample of test data
print("\nPerformance summary (on test data):")
loss, acc = model.evaluate(x_test,  y_test, verbose=2)
print("\tloss = {:5.3f}\n\taccuracy = {:5.3f}".format(loss, acc))

import numpy as np
y = np.argmax(model.predict(x_test), axis=-1)
print("\n\033[1mModel predictions for the first 10 test examples:\033[0m")
for i in range(10):
    print("ground truth for test example {:} = {:}, model prediction = {:}".format( i, y_test[i], y[i]) )

