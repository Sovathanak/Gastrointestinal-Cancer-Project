import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
import numpy as np


# Import extratcted features
IV3_features = pd.read_csv("extractedFeatures\\InceptionV3features.csv")
Res_features = pd.read_csv("extractedFeatures\\ResNet18features.csv")
VGG_features = pd.read_csv("extractedFeatures\\VGG16features.csv")

# Partition data into training and testing sets
# For now only using IV3 - can expland later
IV3_features = IV3_features.to_numpy()
IV3_features = np.append(IV3_features[0:4001, :], IV3_features[-6000:-1, :], axis=0)

X, y = IV3_features[:, 0:9], IV3_features[:, 10]

# Preprocessing test
X = preprocessing.scale(X)

X_train_IV3, X_test_IV3, y_train_IV3, y_test_IV3 = train_test_split(
    X, y, test_size=0.1, train_size=0.9
)

X_train_IV3, X_valid_IV3, y_train_IV3, y_valid_IV3 = train_test_split(
    X_train_IV3, y_train_IV3, test_size=float(1 / 9), train_size=float(8 / 9)
)
print("Data split done.")

# Fit decision tree
dtree = tree.DecisionTreeClassifier()
dtree.fit(X_train_IV3, y_train_IV3)

# Plot fitted tree
tree.plot_tree(dtree)

# Predict the response for test dataset
y_pred = dtree.predict(X_test_IV3)

# Report accuracy
print("Accuracy:", metrics.accuracy_score(y_test_IV3, y_pred))

# Print confusion matrix
cm = metrics.confusion_matrix(y_test_IV3, y_pred)

print("Confusion matrix:")
print(cm)
print(metrics.classification_report(y_test_IV3, y_pred))
