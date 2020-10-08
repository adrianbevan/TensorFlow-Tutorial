This file provides a brief overview of the contets of this directory

api.py
  This is the applicaiton programmer interface - basically a set of getter
  functions for different data samples that the examples use, including:
    - linear data (y=mx+c)
    - parabolic data (y=x^2)
    - Iris data (from sklearn)
    - MNIST data (from keras)
    - CFAR10 data (from keras)

LinearRegression.py
  This is a simple linear regression example
  
NN.py
  This is a multilayer perceptron example for use with the MNIST data
  
NN_parabola.py
  This is a multilayer perceptron example with the parabolic data
  
CNN.py
  This is a convolutional neural network for use with the MNIST and CFAR10
  data.
  
HyperParameter Scan Scripts:
----------------------------
  The following scripts perform a hyperparameter scan to study the accuracy and loss
  as a function of the batch size, drop out fraction, leaky ReLU alpha parameter and
  the validation split fraction, respectively.

    ValidationSplitNN.py
    BatchSizeNN.py
    DropoutNN.py
    LeakyReluScanNN.py
