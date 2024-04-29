import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy
import pandas as pd

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

# Fix random seed for reproducibility


# Load the data set


# Extract the features and target variable


# Split into test and train data sets


# Scale the data


# Create the model


# Compile the model


# Fit the model


# Evaluate the model


