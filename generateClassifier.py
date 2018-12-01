from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd

train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")
X_train, y_train, X_test, y_test = train.iloc[:, 1:].values, train['label'].values, test.iloc[:, 1:].values, test['label'].values
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Extract the hog features
list_hog_fd = []
for feature in X_train:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

clf = LinearSVC()

clf.fit(hog_features, y_train)
print(clf.score(hog_features, y_train))

# Save the classifier
joblib.dump(clf, "digits_cls.pkl", compress=3)