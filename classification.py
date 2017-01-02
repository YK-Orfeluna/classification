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

#FLAG = "f"					# 学習データとテストデータでF値を使って精度検証
FLAG = "cross"				# K分割交差検定を行う（K = CV）

#IRIS = True				# irisデータを使ったデモ
IRIS = False

JOBS = 2					# 同時進行スレッド数

if IRIS :					# irisデータを扱う場合のチューニング変数
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
		row = TRAIN.shape[1]
		x_train = TRAIN[:,:row-1]
		y_train = TRAIN[:,row-1:]

		row = TEST.shape[1]
		x_test = TEST[:,:row-1]
		y_test = TEST[:,row-1:]

elif FLAG == "cross" :		# 交差検定を行う場合の，学習データの作成
	if IRIS :				# irisデータを扱う場合
		x_train = datasets.load_iris().data
		y_train = datasets.load_iris().target
		print x_train.shape
		print y_train.shape
		exit()

	else :
		row1 = TRAIN.shape[1]
		x_train = TRAIN[:,:row1-1]
		y_train = TRAIN[:,row1-1:]
		if len(y_train.shape) == 2 :
			y_train = y_train[:, 0]
		#y_train = y_train.astype(np.int64)
		#print x_train.shape
		#print y_train.shape
		#exit()

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
			param = PARAM_KNN
		elif flag == "liner" :
			est = svm.SVC()
			param = PARAM_LINER
		elif flag == "rbf" :
			est = svm.SVC()
			param = PARAM_RBF
		elif flag == "poly" :
			est = svm.SVC()
			param = PARAM_POLY
		else :
			sys.exit("Only 'knn', 'liner', 'poly' or rbf'")

		if flag == "knn" :
			label += flag
		else :
			label += "SVM_%s" %flag
		
		t = open(label + ".txt", "w")
		self.output(t, label)


		clf = GridSearchCV(est, param, cv=CV, n_jobs=JOBS)
		clf.fit(x_train, y_train)
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
	#app.main("poly")
	app.main("rbf")

	sys.exit("System Exit")