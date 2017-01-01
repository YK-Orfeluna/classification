# -*- coding: utf-8 -*-

import sys 
import numpy as np
from sklearn import datasets, neighbors, svm
from sklearn.model_selection import GridSearchCV

CSV = ","
TSV = "\t"
DEL = CSV

TRAIN = "iris_train.csv"
TRAIN = np.genfromtxt(TRAIN, delimiter=DEL)

TEST = "iris_test.csv"
TEST = np.genfromtxt(TEST, delimiter=DEL)

FLAG = "f"					# 学習データとテストデータでF値を使って精度検証
#FLAG = "cross"				# K分割交差検定を行う（K = CV）

#IRIS = True				# irisデータを使ったデモ
IRIS = False

JOBS = 2					# 同時進行スレッド数

if IRIS :					# irisｓデータを扱う場合のチューニング変数
	C = GAMMA = DEGREE = NEIGHBOR = [1, 2, 3, 4, 5]
else :						# 通常のチューニング変数
	C = np.append(np.array([1e-06, 1e-05, 1e-04, 0.001, 0.01]), np.arange(0.1, 30.1, 0.1))
	GAMMA = np.append(np.array([0, 1e-06, 1e-05, 1e-04, 0.001, 0.01]), np.arange(0.1, 30.1, 0.1))
	DEGREE = np.arange(2, 11)
	NEIGHBOR = np.arange(1, 16)
WEIGHT = ["uniform", "distance"]
CV = 20

PARAM_KNN = {"n_neighbors":NEIGHBOR ,"weights":WEIGHT}
PARAM_LINER = {"kernel":["linear"], "C":C}
PARAM_POLY = {"kernel":["poly"], "C":C, "degree":DEGREE}
PARAM_RBF = {"kernel":["rbf"], "C":C, "gamma":GAMMA}

if FLAG == "f" :			# F値で検定を行う場合の，学習・テストデータの作成
	if IRIS :				# irisデータを扱う場合
		iris = datasets.load_iris()
		x_train = iris.data[xrange(0, len(iris.data), 2)]
		y_train = iris.target[xrange(0, len(iris.data), 2)]
		x_test = iris.data[xrange(1, len(iris.data), 2)]
		y_test = iris.target[xrange(1, len(iris.data), 2)]
		target_names = iris.target_names

	else :
		row1 = TRAIN.shape[1]
		row2 = TEST.shape[1]
		x_train = TRAIN[:,:row1-1]
		y_train = TRAIN[:,row1-1:]
		x_test = TEST[:,:row2-1]
		y_test = TEST[:,row2-1:]

elif FLAG == "cross" :		# 交差検定を行う場合の，学習データの作成
	if IRIS :				# irisデータを扱う場合
		x_train = datasets.load_iris().data
		y_learn = datasets.load_iris().target

	else :
		row1 = TRAIN.shape[1]
		x_train = data[:,:row1-1]
		y_train = data[:,row1-1:]

class App() :
	def __init__(self) :
		self.label = "Classification_"

	def output(self, t, p) :
		t.write(str(p) + "\n")
		print(p)

	def main(self, flag) :
		label = self.label

		if flag == "knn" :
			est = neighbors.KNeighborsClassifier()
			param = {"n_neighbors":neighbor ,"weights":weight}
		elif flag == "liner" :
			est = svm.SVC()
			param = {"kernel":["linear"], "C":c}
		elif flag == "rbf" :
			est = svm.SVC()
			param = {"kernel":["rbf"], "C":c, "gamma":gamma}
		elif flag == "poly" :
			est = svm.SVC()
			param = {"kernel":["poly"], "C":c, "degree":degree}
		else :
			sys.exit("Only 'knn', 'liner', 'rbf'")

		if flag == "knn" :
			label += flag
		else :
			label += "SVM_%s" %flag
		
		t = open(label + ".txt", "w")
		self.output(t, label)


		clf = GridSearchCV(est, param, cv=cv, n_jobs=jobs)
		clf.fit(X_learn, y_learn)
		print("fit")

		b_score = "Best Score: %s" %clf.best_score_
		self.output(t, b_score)
		b_param = "Best Param: %s" %clf.best_params_
		self.output(t, b_param)

		di = clf.cv_results_
		di_params = "Each-Param: %s" %str(di["params"])
		self.output(t, di_params)
		mean = "Each Mean-Score: %s" %di["mean_test_score"]
		self.output(t, mean)
		std = "Each Std-Score: %s" %di["std_test_score"]
		self.output(t, std)

		"""
		for k, v in sorted(di.items()):				# cv_results_の中身をソートしてprint()
			print(str(k) + ": " + str(v) + "\n")
		"""

		t.close()

if __name__ == "__main__" :
	app = App()
	app.main("knn")
	app.main("liner")
	app.main("poly")
	app.main("rbf")

	sys.exit("System Exit")