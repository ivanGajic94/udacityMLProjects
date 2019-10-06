#!/usr/bin/python
"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
import time
sys.path.append("./tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print("the shortened length: ", round(len(features_train) / 100))

#########################################################
### your code goes here ###
clf = svm.SVC(kernel='rbf', C=10000.0)
print('Classifier built, beggining training')
start = time.clock()
clf.fit(features_train, labels_train)
end = time.clock()
print("time to train was: ", end - start)
start = time.clock()
pred = clf.predict(features_test)
end = time.clock()
print("time to predict was: ", end - start)
print("The accuracy is: ", accuracy_score(pred, labels_test))
print("The prediction of index 100 is: ", pred[100])
print("The prediction of index 10 is: ", pred[10])
print("The prediction of index 26 is: ", pred[26])
print("The prediction of index 50 is: ", pred[50])

chris_pred = [i for i in pred if i == 1]
sara_pred = [i for i in pred if i == 0]
#########################################################
