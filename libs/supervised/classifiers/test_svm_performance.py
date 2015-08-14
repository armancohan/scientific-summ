from __future__ import division
from sklearn import svm
from sklearn import cross_validation
from sklearn import datasets
import json
from sklearn import linear_model
with open('/Users/rmn/Downloads/ltr-features-D1418_TRAIN') as mf:
    data = json.load(mf)
print data['x_train'][0]
train = [[x[0], x[1], x[3]] for x in data['x_train']]
x_train, x_test, y_train, y_test = cross_validation.train_test_split(
    train, data['y_train'], test_size=0.2, random_state=0)
clf = svm.SVC(kernel='linear', class_weight={1: 5}, C=1).fit(x_train, y_train)
# clf = linear_model.SGDClassifier(class_weight={1:5}).fit(x_train, y_train)
tp, fp, all = 0, 0, 0
for i, x in enumerate(x_test):
    predict = clf.predict(x)
    if y_test[i] == 1:
        all += 1
    if predict == 1 and y_test[i] == 1:
        tp += 1
    elif predict == 1 and y_test[i] == 0:
        fp += 1
try:
    print tp, fp, all, tp / all, tp / (tp + fp)
except:
    print all
print clf.score(x_test, y_test)
print [x[1] for x in data['x_train']]
'''
Created on Mar 10, 2015

@author: rmn
'''
