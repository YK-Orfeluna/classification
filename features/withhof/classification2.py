# -*- coding: utf-8 -*-


import json
from os.path import splitext, exists, basename
from time import ctime

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals import joblib

#from bayesian_optimizer import BayesianOptimizer


from sys import version_info
if version_info[0] == 2 :
	from commands import getoutput
elif version_info[0] == 3 :
	from subprocess import getoutput


KNN = "KNN"
SVM = "SVM"
Kmeans = "Kmeans"
GMM = "GMM"

CV = "CV"
Bayes = "Bayes"
auto = "auto"

def k(n) :
	out =  (1 + np.log(n) / np.log(2)) * 4
	return int(round(out, 0))

def read_data(filename) :					# ファイル読み込み（csv/tsvに限る）
	if splitext(filename)[1] == ".csv" :
		df = pd.read_csv(filename, header=None, index_col=None)
	elif splitext(filename)[1] == ".tsv" :
		df = pd.read_csv(filename, header=None, index_col=None, delimiter="\t")
	else :
		exit("Error: this script can read CSV or TSV file only.")

	data = df.values[:, :-1]			# 最後の1行以外をデータセットにする
	label = df.values[:, -1]			# 最後の1行をラベルにする

	return data, label

class Classification() :
	def __init__(self, njobs, traindata, testdata, outdir) :

		self.njobs = int(round(njobs, 0))


		self.train_data = np.array([])
		self.train_label = np.array([])
		self.test_data = np.array([])
		self.test_label = np.array([])
		self.gs_data = np.array([])
		self.gs_label = np.array([])

		self.load_dataset(traindata, testdata)


		self.param = {}
		self.method = ""
		self.eval = ""

		self.load_config()


		if outdir[-1] == "/" :
			self.outdir = outdir[: -1]
		else :
			self.outdir = outdir

		if not exists(self.outdir) :
			getoutput("mkdir %s" %self.outdir)
			print("mkdir %s" %self.outdir)


		self.best_clf = None

		self.fd = open("%s/rslt.txt" %outdir, "w")

	def load_config(self) :			# 設定用のjsonファイルを読み込む
		self.method = SVM

		self.cv = 20

		self.eval = CV

		kernel = ["rbf"]
		c = np.append(np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]), np.arange(0.1, 30.1, 0.1)).astype(np.float64)
		g = np.append(np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]), np.arange(0.0, 30.1, 0.1)).astype(np.float64)
		self.param = {"kernel": kernel, "C": c, "gamma": g}

		#print("[%s]done: read %s" %(ctime(), config_json))


	def load_dataset(self, traindata, testdata) :
		if testdata == -1 or testdata == "-1" :								# testdataが-1だった場合，traindataを分割する
			data, label = read_data(traindata)

			self.gs_data = data.copy()
			self.gs_label = label.copy()

			self.train_data = data[0::2, :].copy()
			self.train_label = label[0::2].copy()

			self.test_data = data[1::2, :].copy()
			self.test_label = label[1::2].copy()

			print("[%s]\ndone: read %s, %s" %(ctime(), traindata, testdata))

		else :
			self.train_data, self.train_label = read_data(traindata)		# 学習データの読み込み
			self.test_data, self.test_label = read_data(testdata)			# テストデータの読み込み

			self.gs_data = self.train_data.copy()
			self.gs_label = self.train_data.copy()

			print("[%s]\ndone: read %s" %(ctime(), traindata))

	def crossvalidation(self, param=None, debug=True) :
		if debug :
			print("[%s]\nstart %d-fold Cross Validation\n" %(ctime(), self.cv))


		if param == None :
			param = self.param


		gs = GridSearchCV(self.clf, param, cv=self.cv, n_jobs=self.njobs)
		gs.fit(self.gs_data, self.gs_label)


		gs_result = pd.DataFrame(gs.cv_results_)			# CrossValidationの結果
		if debug :
			gs_result.to_csv("%s/gridsearch.csv" %self.outdir)


		best_param = gs.best_params_						# 最良平均正解率の時のパラメータ
		if debug :
			print("best parameter:\t%s" %best_param)
			self.fd.write("best parameter:\n%s\n" %best_param)

		accuracy = gs.best_score_							# 最良平均正解率
		if debug :
			print("best test accuracy:\t%s" %accuracy)
			self.fd.write("best test accuracy: \t%s\n" %accuracy)

		index = gs.best_index_
		accuracy_SD = gs_result["std_test_score"][index]	# 最良平均正解率の標準偏差（SD）
		
		if debug :
			print("SD of test accuracy:\t%s\n" %accuracy_SD)
			self.fd.write("SD of test accuracy:\t%s\n" %accuracy_SD)

		self.best_clf = gs.best_estimator_					# 最良平均正解率のモデル
		if debug :
			print(self.best_clf, "\n")
			joblib.dump(self.best_clf, "%s/clf.pkl" %self.outdir)			# pklファイルとして分類器を出力する

		return accuracy


	def classification(self) :
		clf = self.best_clf
		clf.fit(self.train_data, self.train_label)

		predict = clf.predict(self.test_data)
		matrix = confusion_matrix(self.test_label, predict)			# 混合行列
		matrix = matrix.astype(np.str)
		report = classification_report(self.test_label, predict)	# 混合行列を基にしたPresicion, Recall, F-measure

		print("confusion matrix:")
		print(matrix, "\n")
		print("Result:")
		print(report)

		np.savetxt("%s/confusion_matrix.csv" %self.outdir, matrix, delimiter=",", fmt="%s")
		
		self.fd.write("%s" %report)

		"""
		reports = report.strip().split()
		precision = float(reports[-4])
		recall = float(reports[-3])
		f1 = float(reports[-2])

		print("Precision, Recall, F-measure: %s, %s, %s" %(precision, recall, f1))
		self.fd.write("Precision: %s\n" %precision)
		self.fd.write("Recall: %s\n" %recall)
		self.fd.write("F-measure: %s\n" %f1)
		"""
	"""
	def bayesian(self) :
		for p1 in self.param :
			bo = BayesianOptimizer(p1)

			for p2 in bo.supply_next_param() :
				y = self.classification(param=p1, debug=False)
				bo.report(y)
			bayes_rslt = bo.best_results()

			print(bayes_rslt)
			self.fd.write(bayes_rslt)
			self.fd.write("\n")
	"""

	def main(self) :
		if self.method == KNN :
			self.clf = KNeighborsClassifier()
		elif self.method == SVM :
			self.clf = SVC(probability=True, decision_function_shape="ovr")
		elif self.method == Kmeans :
			self.clf = KMeans()
		elif self.method == GMM :
			self.clf = GaussianMixture()
		else :
			exit("method")

		if self.eval == CV :
			self.crossvalidation()
			self.classification()

		#elif self.eval == Bayes :
		#	self.bayesian()

		else :
			exit("eval")

		self.fd.close()

def run(traindata) :
	print(traindata)
	outdir = basename(splitext(traindata)[0])
	clf = Classification(1, traindata, "-1", outdir)
	clf.main()

if __name__ == "__main__" :

	from time import sleep

	sleep(64800)
	
	from multiprocessing import cpu_count, Pool
	import glob

	target_names = glob.glob("*.csv")
	print(target_names)
	#run(target_names[0])
	p = Pool(6)
	p.map(run, target_names)

	exit("[%s]\nscript: done" %ctime())
