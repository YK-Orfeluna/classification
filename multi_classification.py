# -*- coding: utf-8 -*-

import sys 
import numpy as np
from sklearn import datasets, neighbors, svm
from sklearn.model_selection import GridSearchCV

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

	def main(self, classification) :
		label = self.label

		if classification == "knn" :
			est = neighbors.KNeighborsClassifier()
			param = {"n_neighbors":neighbor ,"weights":weight}
		elif classification == "liner" :
			est = svm.SVC()
			param = {"kernel":["linear"], "C":c}
		elif classification == "rbf" :
			est = svm.SVC()
			param = {"kernel":["rbf"], "C":c, "gamma":gamma}
		elif classification == "poly" :
			est = svm.SVC()
			param = {"kernel":["poly"], "C":c, "degree":degree}
		else :
			sys.exit("Only 'knn', 'liner', 'rbf'")

		if classification == "knn" :
			label += classification
		else :
			label += "SVM_%s" %classification
		
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