# Last edited by Alex Waddington 30/08/2021  - code partly based on methods written by
# Avinash Navlani https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import random

# Randomly sample data ensuring proper target class distribution


# Import extratcted features
IV3_features = pd.read_csv("extractedFeatures\\InceptionV3features.csv")
Res_features = pd.read_csv("extractedFeatures\\ResNet18features.csv")
VGG_features = pd.read_csv("extractedFeatures\\VGG16features.csv")

# Partition data into training and testing sets
# For now only using IV3 - can expland later
IV3_features = IV3_features.to_numpy()
IV3_features = np.append(IV3_features[0:4001, :], IV3_features[-6000:-1, :], axis=0)

Cancer_lst = []
for i in range(len(IV3_features)):
    if IV3_features[i][-1] == "MSS":
        Cancer_lst.append(0)
    elif IV3_features[i][-1] == "MSIMUT":
        Cancer_lst.append(1)
    else:
        print("Something is broken, somewhere...")

Cancer_lst = np.array(Cancer_lst)
Cancer_lst = Cancer_lst.reshape(-1, 1)

print(Cancer_lst)


X_train_IV3, X_test_IV3, y_train_IV3, y_test_IV3 = train_test_split(
    IV3_features[:, 0:9], Cancer_lst
)

index = int(len(X_train_IV3) * 0.1)
X_valid_IV3, y_valid_IV3 = X_train_IV3[0:index], y_train_IV3[0:index]
X_train_IV3, y_train_IV3 = X_train_IV3[index:], y_train_IV3[index:]

# Perform gridsearch to set the hyperparameters C and gamma
# Utility function to move the midpoint of a colormap to be around
# the values of interest.
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_valid_IV3, y_valid_IV3)

print(
    "The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)

# Fit a model as per the optimal hyperparameters
svm_opt = svm.SVC(C=grid.best_params_["C"], gamma=grid.best_params_["gamma"])
svm_opt.fit(X_train_IV3, y_train_IV3)
predicted = svm_opt.predict(X_test_IV3)
print(predicted)
acc = metrics.accuracy_score(y_test_IV3, predicted)
auc = metrics.auc(y_test_IV3, predicted)
prec = metrics.precision_score(y_test_IV3, predicted)
cm = metrics.confusion_matrix(y_test_IV3, predicted)

# print("Accuracy: {}  AUC: {}  Precision: {}".format(acc, auc, prec))
print("Confusion matrix:")
print(cm)
