# coding: utf-8

try :
	from skopt import gp_minimize
except ImportError :
	exit("You have to install \"scikit-optimize.\"")


from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()

L = "log-uniform"
params = [	[2**-30, 2**30, L],
			[2**-30, 2**30, L], 
			#["linear", "rbf", "poly"],
			#[i for i in range(1, 11)]
			]

def f(x) :
	#C, gamma, kernel, degree = x
	C, gamma = x
	svc = svm.SVC(C=C, gamma=gamma)
	clf = GridSearchCV(svc, dict(), cv=20)
	clf.fit(iris.data, iris.target)

	return clf.best_score_ * -1

res = gp_minimize(f, params, acq_func="EI", n_calls=100, n_jobs=6)
print("Best Parameters\n\tC: %s\n\tgamma: %s" %(res.x[0], res.x[1]))
print("Best Score: %s" %res.fun*-1)
