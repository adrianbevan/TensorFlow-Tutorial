"""
    NN.py  - This is an MLP example using Keras with MNIST data

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
import matplotlib.pyplot as plt
import api

api.PrintGNULicense()

#
# Training configuration
#
ValidationSplit = 0.5
BatchSize       = 20
Nepochs         = 20
DropoutValue    = 0.6

# The MNIST data is a standard data set of hand written numbers,
#     0, 1, 2, 3, 4, 5, 6, 7, 8, 9
# that can be found online at:
#
#    http://yann.lecun.com/exdb/mnist/
#
# We load these in via keras using a tutorial api function that
# also handles the normalisation of the data values
x_train, y_train, x_test, y_test = api.getMNIST()

# these data are 28x28 pixel images and before we can use these
# with a neural network we need to reshape the impage to
# a flat 784 pixel input feature space. This is done
# using the tf.keras.layers.Flatten function.
#
# Once we have done that we can then create a neural network using
# that takes this 784 dimensional input feature space and
# passes it into a neural network with a single layer of nodes
# with a specified activation function (relu in this case),
# and as the MNIST data have 10 possible outputs, we need to specify
# those outputs using the same nomenclature (tf.keras.layers.Dense).
#
# Dropout is a feature discussed in the notes that helps avoid
# coadaption of nodes in a network and overtraining of the data.
#
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
  tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
  tf.keras.layers.Dropout(DropoutValue),
  tf.keras.layers.Dense(10)
])

print("--------------------------------------------------------------------------------------------------------------")
print("Will scan through drop out keep probabilities to explore the model performance as a function of this parameter")
print("--------------------------------------------------------------------------------------------------------------\n\n")
print("Input data MNIST")
print("2 layer MLP with configuration 784:128:128:10")
print("Dropout values       = ", DropoutValue)
print("Leaky relu parameter =  0.1")
print("ValidationSplit      = ", ValidationSplit)
print("BatchSize            = ", BatchSize)
print("Nepochs              = ", Nepochs, "\n")


# now specify the loss function - cross entropy
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


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

print("\nDisplay the evolution of the accuracy as a function of the training epoch")
print("  N(Epochs)        = ", Nepochs)
print("  accuracy (train) = ", history.history['accuracy'])
print("  accuracy (test)  = ", history.history['val_accuracy'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
print("Plotting the output to fig/NN_MLP_accuracy_vs_epochs.*")
plt.savefig("fig/NN_MLP_accuracy_vs_epochs.pdf")
plt.savefig("fig/NN_MLP_accuracy_vs_epochs.png")
plt.clf()

print("\nDisplay the evolution of the loss as a function of the training epoch")
print("  N(Epochs)        = ", Nepochs)
print("  loss (train)     = ", history.history['loss'])
print("  loss (test)      = ", history.history['val_loss'])

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
print("Plotting the output to fig/NN_MLP_loss_vs_epochs.*")
plt.savefig("fig/NN_MLP_loss_vs_epochs.pdf")
plt.savefig("fig/NN_MLP_loss_vs_epochs.png")

# having finished training the model, use this to evaluate the performance on a sample of test data
print("\nPerformance summary (on test data):")
loss, acc = model.evaluate(x_test,  y_test, verbose=2)
print("\tloss = {:5.3f}\n\taccuracy = {:5.3f}".format(loss, acc))
