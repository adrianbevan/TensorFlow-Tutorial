This file provides a brief overview of the contets of this directory

api.py
  This is the applicaiton programmer interface - basically a set of getter
  functions for different data samples that the examples use, including:
    - linear data (y=mx+c)
    - parabolic data (y=x^2)
    - Iris data (from sklearn)
    - MNIST data (from keras)
    - CFAR10 data (from keras)

TensorFlow/Keras Examples
-------------------------
LinearRegression.py
  This is a simple linear regression example. i.e. this is a ML function
  approximation example for the function y = mx+c.

NN.py
  This is a multilayer perceptron example for use with the MNIST data
  
NN_parabola.py
  This is a multilayer perceptron example with the parabolic data, i.e.
  this is a ML function approximation example for the function y=x^2.
  
CNN.py
  This is a convolutional neural network for use with the MNIST and CFAR10
  data.

NN_para_save.py
NN_para_load.py
  This is an extension of the NN_parabola.py script to split out the functionality
  of learning a model (the MLP to approximate y=x^2), and the functionality to recall
  that model in order to make predictions.

HyperParameter Scan Scripts:
----------------------------
  The following scripts perform a hyperparameter scan to study the accuracy and loss
  as a function of the batch size, drop out fraction, leaky ReLU alpha parameter and
  the validation split fraction, respectively.

    ValidationSplitNN.py
    BatchSizeNN.py
    DropoutNN.py
    LeakyReluScanNN.py

SciKit Learn Examples 
---------------------
These have been included to provide access to examples of other types of supervised learning
algorithms that may be encountered in the physcial sciences, beyond neural networks.

SK_DT.py
  This is a decision tree example (uses sklearn)
  
SK_BDT.py
  This is a bosted decision tree example (uses the Adaboost from sklearn)

SK_RF.py
  This is a random forest example (uses sklearn)
  
SK_SVM.py
  This is a support vector machine example (uses the SVM from sklearn, i.e. libsvm)
