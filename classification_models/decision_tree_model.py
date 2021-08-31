import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import pandas as pd
import numpy as np

# Import the csv file 
IV3_features = pd.read_csv("extractedFeatures\\InceptionV3features.csv")
Res_features = pd.read_csv("extractedFeatures\\ResNet18features.csv")
VGG_features = pd.read_csv("extractedFeatures\\VGG16features.csv")