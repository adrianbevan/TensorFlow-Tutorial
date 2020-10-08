"""
    BatchSizeNN.py  - This example optimises the batch size fraction
                    for an MLP model.  Note - this takes a while to run...

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
import time

api.PrintGNULicense()

#
# Training configuration
#
ValidationSplit = 0.5
BatchSizes      = [1, 5, 10, 50, 100, 200, 500, 1000, 30000]
Nepochs         = 50

DropoutValue    = 0.6

# The MNIST data is a standard data set of hand written numbers,
#     0, 1, 2, 3, 4, 5, 6, 7, 8, 9
# that can be found online at:
#
#    http://yann.lecun.com/exdb/mnist/
#
# We load these in via keras
x_train, y_train, x_test, y_test = api.getMNIST()

# now specify the loss function - cross entropy
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

all_loss     = []
all_val_loss = []
all_acc      = []
all_val_acc  = []

print("--------------------------------------------------------------------------------------------------------------")
print("Will scan through batch size hyper parameter to explore the model performance as a function of this")
print("--------------------------------------------------------------------------------------------------------------\n\n")
print("Input data MNIST")
print("2 layer MLP with configuration 784:128:128:10")
print("Dropout value        = ", DropoutValue)
print("Leaky relu parameter = 0.1")
print("ValidationSplit      = ", ValidationSplit)
print("BatchSize            = ", BatchSizes)
print("Nepochs              = ", Nepochs, "\n")
print("N(train)             = ", len(x_train)
print("N(test)              = ", len(x_test)

print("Number of training iterations = ", len(BatchSizes))


for BatchSize in BatchSizes:
  print("\nTraining model using a validation split of ", ValidationSplit)

  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    tf.keras.layers.Dropout(DropoutValue),
    tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    tf.keras.layers.Dropout(DropoutValue),
    tf.keras.layers.Dense(10)
  ])


  # now we can train the model to make predictions.
  #   Use the ADAM optimiser
  #   Specify the metrics to report as accuracy
  #   Specify the loss function (see above)
  # the fit step specifies the number of training epochs
  model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
  print("Training starting at ", time.asctime())
  history  = model.fit(x_train, y_train, validation_split=ValidationSplit, batch_size=BatchSize, epochs=Nepochs)
  print("Training finished at ", time.asctime())

  all_loss.append(history.history['loss'][-1])
  all_val_loss.append(history.history['val_loss'][-1])
  all_acc.append(history.history['accuracy'][-1])
  all_val_acc.append(history.history['val_accuracy'][-1])
  

print("\nDisplay the evolution of the accuracy as a function of the validation sample fraction")
print("  BatchSizes       = ", BatchSizes)
print("  accuracy (train) = ", all_acc)
print("  accuracy (test)  = ", all_val_acc)

plt.plot(BatchSizes, all_acc)
plt.plot(BatchSizes, all_val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Validation sample fraction')
plt.legend(['train', 'test'], loc='upper right')
print("Plotting the output to fig/BatchSizeNN_MLP_accuracy_vs_epochs.*")
plt.savefig("fig/BatchSizeNN_MLP_accuracy.pdf")
plt.savefig("fig/BatchSizeNN_MLP_accuracy.png")
#plt.show()
plt.clf()

print("\nDisplay the evolution of the loss as a function of the validation sample fraction")
print("  BatchSizes       = ", BatchSizes)
print("  loss (train) = ", all_loss)
print("  loss (test)  = ", all_val_loss)

# summarize history for loss
plt.plot(BatchSizes, all_loss)
plt.plot(BatchSizes, all_val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('Validation sample fraction')
plt.legend(['train', 'test'], loc='lower right')
print("Plotting the output to fig/BatchSizeNN_MLP_loss_vs_epochs.*")
plt.savefig("fig/BatchSizeNN_MLP_loss.pdf")
plt.savefig("fig/BatchSizeNN_MLP_loss.png")
#plt.show()

print("\nBatch size scan done\n")
