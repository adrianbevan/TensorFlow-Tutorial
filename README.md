# TensorFlow-Tutorial

                 Machine Learning Tensor Flow Examples

    Copyright (C) 2020 Adrian Bevan, Queen Mary University of London

--------------------------------------------------------------------------
This code provides a set of TensorFlow and Keras tutorials that covers
the following machine learning problems:

* Linear Regression
  - LinearRegression.py:
    Explore the problem of a least squares optimisation, fitting
    to a straight line (y-mx+c), where in this case the uncertainty
    on the data are set to be a relative percentage of the value
    of the input data.
    
* MultiLayer Perceptrons (MLP)
  - NN.py
    Fit a 2 layer MLP to the MNIST data.
    
  - NN_parabola.py
    2 layer MLP fitting a parabolic function (y=x^2)
    
    
* Convolutional Neural Networks (CNNs)
  - CNN.py:
    CNN fitting MNIST or CFAR10 data.  To fit the MNIST data ensure that
        UseMNIST = True
    is set.  In order to fit CFAR10 data then set UseMNIST to False.

--------------------------------------------------------------------------
Each of the examples has been written to allow the user to explore
pedegogical aspects of machine learning, starting with probing the
performance of optimisation via the linear regression example, through
to training performance via test vs train loss function and accuracy as
a function of the training epochs.

These examples are accompanied by a set of notes that indicate suggested
exercises using these examples in order to build a deeper understanding
of the methods, their pathologies and some ways that will allow users
to get hands on experience of some of the basics related to machine
learining.

If you find these useful then you might be interested in looking at some
of my machine learning-related teaching materials that can be found
online at:

          https://pprc.qmul.ac.uk/~bevan/teaching.html

--------------------------------------------------------------------------

Note that requirements.txt lists specific package versions that this tutorial 
has been written with.  That file has been included to ensure that this
repository can be used with Binder (see https://mybinder.org for details).

AB 8th Oct 2020
