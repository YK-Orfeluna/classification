# -*- coding: utf-8 -*-

import sys 
import numpy as np
from sklearn import datasets, neighbors, svm
from sklearn.model_selection import GridSearchCV

CSV = ","
TSV = "\t"
DEL = CSV

TRAIN = ""
#TRAIN = np.genfromtxt(TRAIN, delimiter=DEL)

TEST = ""
#TEST = np.genfromtxt(TEST, delimiter=DEL)

FLAG = "f"
#FLAG = "cross"

IRIS = True
#IRIS = False

JOBS = 2

C = np.append(np.array([1e-06, 1e-05, 1e-04, 0.001, 0.01]), np.arange(0.1, 30.1, 0.1))
GAMMA = np.append(np.array([0, 1e-06, 1e-05, 1e-04, 0.001, 0.01]), np.arange(0.1, 30.1, 0.1))
DEGREE = np.arange(2, 11)
NEIGHBOR = np.arange(1, 16)
WEIGHT = ["uniform", "distance"]
CV = 20

if FLAG == "f" :
	if IRIS :
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

elif FLAG == "cross" :
	if IRIS :
		x_train = datasets.load_iris().data
		y_learn = datasets.load_iris().target

	else :
		row1 = TRAIN.shape[1]
		x_train = data[:,:row1-1]
		y_train = data[:,row1-1:]

exit()


argvs = sys.argv								# コマンドライン引数を取得する
if len(argvs) == 3 :							# 引数の数が適切な時（3）
	filename = argvs[1] + ".csv"				# 1番目の引数をcsvファイルの名前として取り扱う（ただし".csv"は入力不要）
	print(filename)
	data = np.genfromtxt(filename, delimiter=',')
	col, row = data.shape
	X_learn = data[:,:row-1]
	y_learn = data[:,row-1:]

	# チューニング用の変数を作る
	c = np.append(np.array([1e-06, 1e-05, 1e-04, 0.001, 0.01]), np.arange(0.1, 30.1, 0.1))
	gamma = np.append(np.array([0, 1e-06, 1e-05, 1e-04, 0.001, 0.01]), np.arange(0.1, 30.1, 0.1))
	degree = np.arange(2, 11)
	neighbor = np.arange(1, 16)

	jobs = int(argvs[2])						# 2番目の引数を並行処理の数として取り扱う（コア数/スレッド数を超えても無駄）

else :											# コマンドライン引数が適切に入力されなかった場合，irisデータを使ったテストを行う
	print("To Use 'Iris-Data'")
	X_learn = datasets.load_iris().data
	y_learn = datasets.load_iris().target

	c = gamma = degree = neighbor = [1, 2, 3]	# チューニング用の変数を作る
	jobs = 2									# 並行処理の同時並行数

weight = ["uniform", "distance"]				# knnの変数
cv = 20											# グリッドサーチの回数


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