from sklearn import metrics
from sklearn import svm
import random
import numpy as np

from sklearn.svm import SVC

# fit a SVM model to the data

size = 5
x1 = np.random.normal(1, 6, size)
x2 = np.random.normal(100, 6, size)

x_val = np.concatenate((x1,x2))


x_val = [np.random.normal(1, 6, size)] * 5
x_val1 = [np.random.normal(100, 6, size)] * 5

x_val = np.concatenate((x_val,x_val1))
print(x_val)

y_val = ['類別1'] * size + ['類別2'] * size

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(x_val, y_val) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

predictX = np.random.normal(1, 6, size)
predictX = [100] * size
print('predictX',predictX)
print('predict result:',clf.predict(predictX))

