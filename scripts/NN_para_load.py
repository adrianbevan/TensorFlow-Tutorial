"""
    NN_para_load.py  - This is example of recalling model learned using Keras
                       (for the function y=x^2) and applying that saved model.

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
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True    # we want to use tex formatting for figures
import matplotlib.pyplot as plt
import os
import random

#
# Load the model
#  - note that the leaky_relu activation function used in the NN_para_save script needs
#    to be specified as a custom object in order to be recalled correctly.
#
print("\033[1mLoad the saved model\033[0m")
model = tf.keras.models.load_model('model', custom_objects={'leaky_relu': tf.nn.leaky_relu})
print("Loaded model from disk")

# now specify the loss function to compile the model in order to use it
loss_fn = tf.keras.losses.MeanSquaredError()

# now we can train the model to make predictions.
#   Use the ADAM optimiser
#   Specify the loss function (see above)
model.compile(optimizer='adam', loss=loss_fn)

#
# Having recalled and compiled the model - the task is now to use this to make predictions; 
# so we want to generate some toy input data to work with.
#
xmin  = -10.0
xmax  = 10.0
r = xmax - xmin
x = []
for i in range(1000):
   x.append( r * random.random() + xmin )

#
# Use the model to make predictions
#
y = model.predict(x)

#
# Display those predictions
#
plt.plot(x, y, '.')
plt.title('model prediction')
plt.ylabel('$y=x^{2}$')
plt.xlabel('$x$')
#plt.show()
plt.savefig("fig/NN_fn_approx_loadedmodel_predictions.pdf")
plt.savefig("fig/NN_fn_approx_loadedmodel_predictions.png")
