from typing import List, Any
import pandas as pd
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import csv
import timeit


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
import pickle
import time
import sklearn.model_selection as model_selection
from sklearn.model_selection import ShuffleSplit

histogram_array = np.loadtxt('/home/jo/PycharmProjects/Perception/desc/Codebooks/histogram_array.out')
#print(histogram_array.shape)
histogram_classes = np.loadtxt('/home/jo/PycharmProjects/Perception/desc/Codebooks/histogram_classes.out')
#print(histogram_classes.shape)

X =histogram_array
y = histogram_classes.astype(int)

#print(type(y[0]))
#print(y)

cv = 10
c_parameter = list(np.logspace(-3,3,50))
gamma_parameter = list(np.logspace(-3,3,50))
degree_parameter = [3,4,5,6]
#c_parameter = [1,2,3]
#gamma_parameter = [1,2,3]
#degree_parameter = [2,3,4,5,6,7,8]




start = timeit.default_timer()

parameters = {'kernel':('linear','rbf','sigmoid'), 'C':c_parameter, 'gamma':gamma_parameter}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters,cv=cv, n_jobs=-2,return_train_score=True, verbose=2)
clf.fit(histogram_array,histogram_classes)
clf.cv_results_.keys()

stop = timeit.default_timer()
print('Time', stop - start)



pandasDataFrame = pd.DataFrame(clf.cv_results_)
pandasDataFrame.to_csv('test.csv')

"""
cv_splits =100
c_value_list= np.logspace(-4,5,50)
kernels = ['rbf','sigmoid','linear']
for k in kernels:
    #print(k)
    score_list =[]
    for ci in c_value_list:
        clf = SVC(kernel=k,C=1)
        cv = ShuffleSplit(n_splits=cv_splits, test_size=0.2)
        scores = cross_val_score(clf, X, y, cv=cv)
        score_list.append(scores.mean())
        #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2),"at c =",np.round(ci,4))

    print("kenel =",k," Score =", np.round(max(score_list),2)," c-value= ", np.round(c_value_list[np.argmax(score_list)],4))

#print(clf.score(X_test,y_test))
"""


#pickle.dump(clf, open('/home/jo/PycharmProjects/Perception/SURF/SVM_Models/SVM_model.out', 'wb'))
