"""
    NN_parabola.py  - This is an MLP example using Keras to learn the function y=x^2

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
import matplotlib
matplotlib.rcParams['text.usetex'] = True    # we want to use tex formatting for figures
import matplotlib.pyplot as plt
import api

#
# Training configuration
#
ValidationSplit = 0.5
BatchSize       = 100
Nepochs         = 100
DropoutValue    = 0.2

# load the paraboloc data using the api interface
x_train, y_train, x_test, y_test = api.getParabolicData(xmin=-10, xmax=10, Ntrain=2000, Ntest=1000, Noise=0.05)

#
# This MLP has one input feature, with one output prediction. There
# are two configurable hiddel layers.
#
# Dropout is a feature discussed in the notes that helps avoid
# coadaption of nodes in a network and overtraining of the data.
#
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, input_dim=1, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
  tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
  tf.keras.layers.Dropout(DropoutValue),
  tf.keras.layers.Dense(1)
])

print("--------------------------------------------------------------------------------------------------------------")
print("Will train a multilayer perceptron using some toy data following y = x^2")
print("--------------------------------------------------------------------------------------------------------------\n\n")
print("Input data MNIST")
print("2 layer MLP with configuration 1:128:128:1")
print("Dropout values       = ", DropoutValue)
print("Leaky relu parameter = 0.1")
print("ValidationSplit      = ", ValidationSplit)
print("BatchSize            = ", BatchSize)
print("Nepochs              = ", Nepochs, "\n")
print("N(train)             = ", len(x_train))
print("N(test)              = ", len(x_test))


# now specify the loss function 
loss_fn = tf.keras.losses.MeanSquaredError()

# now we can train the model to make predictions.
#   Use the ADAM optimiser
#   Specify the metrics to report as accuracy
#   Specify the loss function (see above)
# the fit step specifies the number of training epochs
model.compile(optimizer='adam', loss=loss_fn)
history  = model.fit(x_train, y_train, validation_split=ValidationSplit, batch_size=BatchSize, epochs=Nepochs)

# Print out the history keys expected are:
#    loss        The loss function evaluated at each epoch for the training set
#    acc         The accuracy evaluated at each epoch for the training set
#    val_loss    The loss evaluated at each epoch for the validation set
#    val_acc     The accuracy evaluated at each epoch for the validation set
# The val_* entries exist only if there is a validation_split specified
print("history keys = ", history.history.keys())

print("Display the evolution of the loss as a function of the training epoch")
print("  N(Epochs)        = ", Nepochs)
#print("  loss (train)     = ", history.history['loss'])
#print("  loss (test)      = ", history.history['val_loss'])

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
#plt.show()
plt.savefig("fig/NN_fn_approx_loss_vs_epochs.pdf")
plt.savefig("fig/NN_fn_approx_loss_vs_epochs.png")
plt.clf()

# having finished training the model, use this to evaluate the performance on an
# independent sample of test data
loss = model.evaluate(x_test,  y_test, verbose=2)
print("loss = {:5.3f}".format(loss))

#
# use the model to make predictions based on the unseen test data
#
y_predict = model.predict(x_test)
plt.plot(x_test, y_predict, "r.")
plt.plot(x_test, y_test, "b.")
plt.title('model prediction')
plt.ylabel('$y=x^{2}$')
plt.xlabel('$x$')
plt.savefig("fig/NN_fn_approx_predictions.pdf")
plt.savefig("fig/NN_fn_approx_predictions.png")
plt.clf()

delta = []
deltapc = []
for i in range(len(y_predict)):
    thedelta = y_predict[i]-y_test[i]
    delta.append( thedelta )
    if( x_test[i] ):
       deltapc.append( thedelta /  x_test[i] )
    else:
       deltapc.append( 0.0 )

print("Ntest = ", len(delta))
print("Ntest = ", len(x_test))
plt.plot(x_test, delta, "b.")
plt.plot(x_test, deltapc, "r.")
plt.legend(['$\Delta_y$', '$\Delta_y$ (fraction)'], loc='upper right')
plt.title('model prediction accuracy')
plt.ylabel('$\widehat{y}-y$')
plt.xlabel('$x$')
plt.ylim(-10, 10)

plt.savefig("fig/NN_fn_approx_accuracy.pdf")
plt.savefig("fig/NN_fn_approx_accuracy.png")

print("done")
