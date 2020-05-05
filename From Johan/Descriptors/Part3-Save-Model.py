from typing import List, Any
import pandas as pd
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import csv
import timeit
from sklearn.model_selection import train_test_split


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
import pickle
import time
import sklearn.model_selection as model_selection
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix

histogram_array = np.loadtxt('/home/jo/PycharmProjects/Perception/desc/Codebooks/histogram_array.out')
#print(histogram_array.shape)
histogram_classes = np.loadtxt('/home/jo/PycharmProjects/Perception/desc/Codebooks/histogram_classes.out')
#print(histogram_classes.shape)

X =histogram_array
y = histogram_classes.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



#print(type(y[0]))
#print(y)


c_parameter = 0.001
gamma_parameter = 1



start = timeit.default_timer()


clf = svm.SVC(kernel='linear',C=0.001)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))


stop = timeit.default_timer()
print('Time', stop - start)




pickle.dump(clf, open('/home/jo/PycharmProjects/Perception/desc/SVM_Models/SVM_model.out', 'wb'))
