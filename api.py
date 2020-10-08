"""
    api.py  - This is an application programmer interface for a machine learning tutorial
              See the README for more details, including a github location.

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




This is the application programmer interface for this TensorFlow/Keras tutorial.
This file contains useful helper functions to provide different data samples that 
have been written for this tutorial.

  getCFAR10        - load the CFAR10 data from keras as sets of test and train x and y values
  getIrisData      - load the Iris data from scikit-learn as a Bunch
  getLinearData    - generate a sample of data for y = mx+c
  getMNIST         - load the MNIST data from keras as sets of test and train x and y values
  getParabolicData - generate a sample of data for the function y=x^2

"""
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn 
import numpy as np
import random
import sys

def PrintGNULicense():
  """
Print out the GNU preamble
  """
  print("--------------------------------------------------------------------")
  if sys.argv[0] == '':
     print("TensorFlow Machine Learning Tutorial Copyright (C) 2020 Adrian Bevan")
  else:
     print(sys.argv[0], "  Copyright (C) 2020 Adrian Bevan")
  print("\nThis program comes with ABSOLUTELY NO WARRANTY.")
  print("This is free software, and you are welcome to redistribute it")
  print("under certain conditions.")
  print("--------------------------------------------------------------------")
    

#--------------------------------------------------------------------
def getCFAR10():
  """
Get the CFAR10 data. This data set is a standard data set of types of
image:
      'airplane', 'automobile', 'bird', 'cat', 'deer', 
      'dog', 'frog', 'horse', 'ship', 'truck'

that can be found online at:

    https://www.cs.toronto.edu/~kriz/cifar.html

CFAR10 is a subset of CFAR100.  Each example is an image that contains 28x28 
pixels, and has a color channel number of 3 (r,g,b).

arguments:
    None

return values:
    x_train   Training data images 
    y_train   Training data labels (true values corresponding to the images)
    x_test    Test data images
    y_test    Test data labels (true values corresponding to the images)
"""

  print("Loading the CFAR10 data from keras")
  # Load the CFAR10 data via the tensorflow keras dataset:
  cifar10 = tf.keras.datasets.cifar10
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()

  """
  Before we can use these images we need to normalise the
  input ranges to facilitate training. This follows the standard
  recommendations that you will find in references such as
  Yan LeCun's Efficient Backprop: 
  
     http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf 
  
  8 bit greyscale is a number in the range [0, 255], and so we
  divide the data by 255 to map the domain of each of the pixel values
  to [0, 1]
  """
  x_train, x_test = x_train / 255.0, x_test / 255.0

  # return the normalised data
  return x_train, y_train, x_test, y_test




#--------------------------------------------------------------------
def getIrisData():
  """
Get the Iris data used in the the R. A. Fisher paper on linear disciminants
from the scikit-learn package.  

arguments:
  None

returns:
  Bunch    The scikit-learn data set Bunch object that has the following keys:
       data          - a numpy array with the shape (N, M) corresponiding to the data. N corresponds to the number
                       of examples and M the number of features in the data (4 in this case)
       target        - numpy array with shape (N) corresponding to the target values of the data
       feature_names - numpy array with shape (M) containing the list of feature names
       filename      - the file name containing the data
       DESCR         - this is a description of the data
  """
  print("Loading the Iris data from sklearn")
  Bunch = ds.load_iris(return_X_y=False)

  return Bunch







#--------------------------------------------------------------------
def getLinearData(xmin=-10, xmax=10, Ntrain=2000, Ntest=1000, m=1.0, c=0.0, Noise=0.5):
  """
Generate a sample of data following y=mx+c over the specified domain. The
noise value specified is the fraction of noise on y in order to replicate
reality that data are noisy and that we have to learn to approximate a
function given the noise in measurements.

arguments:
    xmin      Minimum value in x to sample
    xmax      Maximum value in x to sample
    Ntrain    Number of train data to generate
    Ntest     Number of test data to generate
    m         gradient for the line
    c         constant offset
    Noise     (fractional) Noise level to generate

return values:
    x_train   Training data 
    y_train   Training data true target values
    x_test    Test data 
    y_test    Test data true target values

Example:
  Generate test and train samples using 5% noise as a function of x, and plot the output

import api
import matplotlib.pyplot as plt
x_train, y_train, x_test, y_test = api.getLinearData(-10, 10, 100, 100, 1.0, 0.1, 0.5)
plt.plot(x_train, y_train, 'r.')
plt.show()
plt.plot(x_test, y_test, 'b.')
plt.show()

  """
  print("Generating the parabola data set")
  x_train = []
  y_train = []
  x_test  = []
  y_test  = []

  #--------------------------------------------------------------------
  def sim_line(xmin, xmax, m, c, Noise):
      """
      Function to simulate a random data point for a parabola
      """
      x = random.random()*(xmax-xmin)+xmin
      y = (m*x+c)*(1 + random.random()*Noise)
    
      return x, y
  #--------------------------------------------------------------------
  
  for i in range( Ntrain ):
    x,y = sim_line(xmin, xmax, m, c, Noise)
    x_train.append(x)
    y_train.append(y)

  for i in range( Ntest ):
    x,y = sim_line(xmin, xmax, m, c, Noise)
    x_test.append(x)
    y_test.append(y)
    
  return x_train, y_train, x_test, y_test






#--------------------------------------------------------------------
def getMNIST():
  """
Get the MNIST data. The MNIST data is a standard data set of hand written numbers,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9
that can be found online at:

    http://yann.lecun.com/exdb/mnist/

Each example is an image that contains 28x28 pixels. This set of 784 features
has a shape (28,28) that can be processed easily using a convolutional network
layer, or can be flattened in order to pass the example into a dense layer

arguments:
    None

return values:
    x_train   Training data images 
    y_train   Training data labels (true values corresponding to the images)
    x_test    Test data images
    y_test    Test data labels (true values corresponding to the images)
"""

  print("Loading the MNIST data from keras")
  # Load the MNIST data via the tensorflow keras dataset:
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  """
  Before we can use these greyscale images we need to normalise the
  input ranges to facilitate training. This follows the standard
  recommendations that you will find in references such as
  Yan LeCun's Efficient Backprop: 
  
     http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf 
  
  8 bit greyscale is a number in the range [0, 255], and so we
  divide the data by 255 to map the domain of each of the pixel values
  to [0, 1]
  """
  x_train, x_test = x_train / 255.0, x_test / 255.0

  # return the normalised data
  return x_train, y_train, x_test, y_test
  


#--------------------------------------------------------------------
def getParabolicData(xmin=-10, xmax=10, Ntrain=2000, Ntest=1000, Noise=0.5):
  """
Generate a sample of data following y=x^2 over the specified domain. The
noise value specified is the fraction of noise on y in order to replicate
reality that data are noisy and that we have to learn to approximate a
function given the noise in measurements.

arguments:
    xmin      Minimum value in x to sample
    xmax      Maximum value in x to sample
    Ntrain    Number of train data to generate
    Ntest     Number of test data to generate
    Noise     Noise level to generate

return values:
    x_train   Training data 
    y_train   Training data true target values
    x_test    Test data 
    y_test    Test data true target values

Example:
  Generate test and train samples using 5% noise as a function of x, and plot the output

import api
import matplotlib.pyplot as plt
x_train, y_train, x_test, y_test = api.getParabolicData(-10, 10, 100, 100, 0.5)
plt.plot(x_train, y_train, 'r.')
plt.show()
plt.plot(x_test, y_test, 'b.')
plt.show()

  """
  print("Generating the parabola data set")
  x_train = []
  y_train = []
  x_test  = []
  y_test  = []

  #--------------------------------------------------------------------
  def sim_parabola(xmin, xmax, Noise):
      """
      Function to simulate a random data point for a parabola
      """
      x = random.random()*(xmax-xmin)+xmin
      y = x*x*(1+random.random()*Noise)
    
      return x, y
  #--------------------------------------------------------------------
  
  for i in range( Ntrain ):
    x,y = sim_parabola(xmin, xmax, Noise)
    x_train.append(x)
    y_train.append(y)

  for i in range( Ntest ):
    x,y = sim_parabola(xmin, xmax, Noise)
    x_test.append(x)
    y_test.append(y)
  # convert to nparrays before returning
  X_test  = np.array(x_test)
  Y_test  = np.array(y_test)
  X_train = np.array(x_train)
  Y_train = np.array(y_train)
  
  return X_train, Y_train, X_test, Y_test



#--------------------------------------------------------------------
# end of api.py
#--------------------------------------------------------------------
