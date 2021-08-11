"""This is Alex's implementation of PCA. Some code adapted from FIT3181 Tutorial 2"""
# Load required packages
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# Import images
x_MSS = []
x_MSIMIUT = []
y_MSS = []
y_MSIMUIT = []

