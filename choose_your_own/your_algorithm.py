#!/usr/bin/python

import matplotlib.pyplot as plt
sys.path.append("./choose_your_own/")
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn import svm

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [
    features_train[ii][0] for ii in range(0, len(features_train))
    if labels_train[ii] == 0
]
bumpy_fast = [
    features_train[ii][1] for ii in range(0, len(features_train))
    if labels_train[ii] == 0
]
grade_slow = [
    features_train[ii][0] for ii in range(0, len(features_train))
    if labels_train[ii] == 1
]
bumpy_slow = [
    features_train[ii][1] for ii in range(0, len(features_train))
    if labels_train[ii] == 1
]

#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################

### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary

# Tuned best output: 0.924
ada_boost = AdaBoostClassifier()
ada_boost.fit(features_train, labels_train)
ada_boost.score(features_test, labels_test)

# Tuned best performance 0.924
decision_trees = tree.DecisionTreeClassifier(min_samples_split=20,
                                             min_samples_leaf=15)
decision_trees.fit(features_train, labels_train)
decision_trees.score(features_test, labels_test)

# Tuned best performance 0.956 rbf , 50000000
support_vector_machines = svm.SVC(kernel='sigmoid', C=100.0)
support_vector_machines.fit(features_train, labels_train)
support_vector_machines.score(features_test, labels_test)

try:
    prettyPicture(support_vector_machines, features_test, labels_test)
except NameError:
    pass
