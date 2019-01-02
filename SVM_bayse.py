# coding: utf-8

try :
	from skopt import gp_minimize
except ImportError :
	exit("You have to install \"scikit-optimize.\"")

import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV, train_test_split

iris = datasets.load_iris()

L = "log-uniform"
params = [	[2**-30, 2**30, L],
			[2**-30, 2**30, L], 
			#["linear", "rbf", "poly"],
			#[i for i in range(1, 11)]
			]

def f(x) :
	global clf
	#C, gamma, kernel, degree = x
	C, gamma = x
	svc = svm.SVC(C=C, gamma=gamma)
	clf = GridSearchCV(svc, dict(), cv=20)
	clf.fit(iris.data, iris.target)

	return clf.best_score_ * -1

def f1(x) :
	global clf
	C, gamma = x
	clf = svm.SVC(C=C, gamma=gamma, probability=True)

	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	return score * -1

def f2(x) :
	global clf
	C, gamma = x
	clf = svm.SVC(C=C, gamma=gamma, probability=True)

	clf.fit(X_train, y_train)

	scores = clf.predict_proba(X_test)

	answer = np.zeros_like(scores)
	for x, i in enumerate(y_test) :
		answer[x, i] = 1.0

	#mean squared error
	#error = 0.5 * np.sum((answer-scores)**2)

	# cross entropy error
	delta = 1e-7
	error = -np.sum(answer*np.log(scores+delta))

	return error

if __name__ == "__main__" :
	clf = svm.SVC()

	N = 100
	J = 8

	X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5)

	res1 = gp_minimize(f1, params, acq_func="EI", n_calls=N, n_jobs=J)
	res2 = gp_minimize(f2, params, acq_func="EI", n_calls=N, n_jobs=J)

	print("Best Parameters\n\tC: %s\n\tgamma: %s" %(res1.x[0], res1.x[1]))
	print("Best Score: %s" %(res1.fun*-1))

	print("---------------------------------------------")

	print("Best Parameters\n\tC: %s\n\tgamma: %s" %(res2.x[0], res2.x[1]))
	print("Best Score: %s" %clf.score(X_test, y_test))
	print("Minimum Error: %s" %res2.fun)

