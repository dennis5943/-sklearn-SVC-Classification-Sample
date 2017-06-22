from sklearn import metrics
from sklearn import svm
import random
import numpy as np

from sklearn.svm import SVC

# fit a SVM model to the data

size = 10000
vector_size = 5

x_val0 = [np.random.normal(1, 6, vector_size) for i in range(0,size)]
x_val1 = [np.random.normal(100, 6, vector_size) for i in range(0,size)]
x_val2= [np.random.normal(200, 6, vector_size) for i in range(0,size)]

x_val = np.concatenate((x_val0,x_val1,x_val2))
#print(x_val)

y_val = ['類別1'] * size + ['類別2'] * size + ['類別3'] * size

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(x_val, y_val) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

#開訓結束，驗證一下

predictX = np.random.normal(1, 6, vector_size)
predictX = [100] * vector_size
print('predictX',predictX)
print('predict result:',clf.predict(predictX))
