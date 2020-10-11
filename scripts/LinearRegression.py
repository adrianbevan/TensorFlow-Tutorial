"""
    LinearRegression.py  - This is a linear regregssion example using Keras

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



------------------------------------------------------------------------------
LinearRegression example using a Keras model. This is a simple example 
		 y = mx+c
fitting example.
"""
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import matplotlib.pyplot as plt
import api

#
# Training configuration
#
Nepochs         = 150

minx = -10
maxx = 10
"""
getLinearData arguments are
    xmin      Minimum value in x to sample
    xmax      Maximum value in x to sample
    Ntrain    Number of train data to generate
    Ntest     Number of test data to generate
    m         gradient for the line
    c         constant offset
    Noise     (fractional) Noise level to generate
"""
x_train, y_train, x_test, y_test = api.getLinearData(minx, maxx, 1000, 1000, 1.0, 0.0, 0.1)

model = tf.keras.models.Sequential()
model.add( tf.keras.layers.Dense(1, activation='linear', input_shape=[1,]) )

print("--------------------------------------------------------------------------------------------------------------")
print("Will perform a linear regression optimisation")
print("--------------------------------------------------------------------------------------------------------------\n\n")
print("Input data Linear Regression Data")
print("Nepochs              = ", Nepochs, "\n")
print("N(train)             = ", len(x_train))
print("N(test)              = ", len(x_test))

# now specify the loss function - cross entropy
loss_fn = tf.keras.losses.MSE

# now we can train the model to make predictions.
#   Use the ADAM optimiser
#   Specify the metrics to report as accuracy
#   Specify the loss function (see above)
# the fit step specifies the number of training epochs
model.compile(optimizer='adam', loss=loss_fn )
history  = model.fit( x_train, y_train, epochs=Nepochs)

# Print out the history keys expected are:
#    loss        The loss function evaluated at each epoch for the training set
#    acc         The accuracy evaluated at each epoch for the training set
print("history keys = ", history.history.keys())

print("Display the evolution of the loss as a function of the training epoch")
print("  N(Epochs)        = ", Nepochs)
print("  loss (train)     = ", history.history['loss'])

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.show()
plt.savefig("fig/LinearRegression_loss_vs_epochs.pdf")
plt.savefig("fig/LinearRegression_loss_vs_epochs.png")
plt.clf()

# plot the model for the data
# we are interested in the first layer
layer = model.layers[0]
weights, biases = layer.get_weights()
print("m = ", weights[0][0])
print("c = ", biases[0])
print("MSE loss = ", history.history['loss'][-1])
miny = weights[0][0]*minx+biases[0]
maxy = weights[0][0]*maxx+biases[0]

print("Fitted line xrange = {:}, y range = {:}".format([minx, maxx], [miny, maxy]))

plt.plot(x_test, y_test, "r.")
plt.plot([minx, maxx], [miny, maxy], "b-")  # plot the fitted line
plt.ylabel('y')
plt.xlabel('x')
plt.legend("test data", "fitted model")
plt.savefig("fig/LinearRegression_xy.pdf")
plt.savefig("fig/LinearRegression_xy.png")

