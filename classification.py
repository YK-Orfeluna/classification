# -*- coding: utf-8 -*-


import json
from os.path import splitext

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals import joblib

from bayesian_optimizer import BayesianOptimizer

KNN = "KNN"
SVM = "SVM"
Kmeans = "Kmeans"
GMM = "GMM"

CV = "CV"
Bayes = "Bayes"
auto = "auto"

import math
def k(n) :
	out =  (1 + math.log(n) / math.log(2)) * 4
	out = int(round(out, 0))
	return out

def read_data(filename) :
	if splitext(filename)[1] == ".csv" :
		df = pd.read_csv(filename, header=None, index_col=None)
	elif splitext(filename)[1] == ".tsv" :
		df = pd.read_csv(filename, header=None, index_col=None, delimiter="\t")
	else :
		exit("Error: this script can read CSV or TSV file only.")

	data = df.values[:, :-1].copy()			# 最後の1行以外をデータセットにする
	label = df.values[:, -1].copy()			# 最後の1行をラベルにする

	return data, label

class Classification() :
	def __init__(self, njobs, config_json, traindata, testdata, rslt) :

		self.njobs = int(round(njobs, 0))


		self.param = {}
		self.method = ""
		self.eval = ""

		self.load_config(config_json)


		self.train_data = np.array([])
		self.train_label = np.array([])
		self.test_data = np.array([])
		self.test_label = np.array([])
		self.gs_data = np.array([])
		self.gs_label = np.array([])

		self.load_dataset(traindata, testdata)


		self.rslt = rslt
		self.fd = open(rslt+".txt", "w")


		self.best_clf = None

	def load_config(self, config_json) :
		if splitext(config_json)[1] == ".json" :
			with open(config_json, "r") as fd:
				config = json.load(fd)			# config用のjsonを読み込む
		else :
			exit("Error: this script can read JSON-file as config-file.\nYou should choose JSON-file.")


		self.method = config["method"]		# 分類手法の読み込み

		if self.method != SVM and self.method != KNN and self.method != Kmeans and self.method != GMM :
			exit("Error: your chose method('%s') is not supported by this sciprt.\nYou should choose '%s' or '%s' or '%s' or '%s'.\nYou chose %s" \
				%(self.method, SVM, KNN, Kmeans, GMM))


		# K-fold Cross ValidationのKを読み込み
		if config["K"] == auto :
			self.cv = k(self.gs_data.shape[0])
		else :
			self.cv = int(round(config["K"], 0))

		if config["evaluation"] == CV or config["evaluation"] == Bayes :
			self.eval = config["evaluation"]
		else :
			exit("Error: your chose CV-method('%s') is not supported by this script\nYou should choose '%s' or '%s'" \
				%(config["evaluation"], CV, Bayes))

		self.param = config["param"]

		print("done: read config-file")


	def load_dataset(self, traindata, testdata) :
		if testdata != "-1" :
			self.train_data, self.train_label = read_data(traindata)		# 学習データの読み込み
			self.test_data, self.test_label = read_data(testdata)			# テストデータの読み込み

			self.gs_data = self.train_data.copy()
			self.gs_label = self.train_data.copy()

		else :															# testdataが-1だった場合，traindataを分割する
			data, label = read_data(traindata)

			self.gs_data = data.copy()
			self.gs_label = label.copy()

			self.train_data = data[0::2, :].copy()
			self.train_label = label[0::2].copy()

			self.test_data = data[1::2, :].copy()
			self.test_label = label[1::2].copy()

		print("done: read datasets")

	def crossvalidation(self, param=None, debug=True) :
		if debug :
			print("start CrossValidation")

		if param == None :
			param = self.param


		gs = GridSearchCV(self.clf, param, cv=self.cv, n_jobs=self.njobs)
		gs.fit(self.gs_data, self.gs_label)


		gs_result = pd.DataFrame(gs.cv_results_)
		if debug :
			gs_result.to_csv("%s_GS.csv" %self.rslt)


		best_param = gs.best_params_
		if debug :
			self.fd.write("best parameter:\n%s\n" %best_param)
			print("best parameter: %s" %best_param)

		accuracy = gs.best_score_
		if debug :
			self.fd.write("best accuracy: %s\n" %accuracy)
			print("best accuracy: %s" %accuracy)

		index = gs.best_index_
		accuracy_SD = gs_result["std_test_score"][index]
		
		if debug :
			self.fd.write("SD of accuracy: %s\n" %accuracy_SD)
			print("SD of accuracy: %s\n" %accuracy_SD)

		self.best_clf = gs.best_estimator_
		if debug :
			print(self.best_clf)
			joblib.dump(self.best_clf, "%s.pkl" %self.rslt)			# pklファイルとして分類器を出力する

		return accuracy


	def classification(self) :
		clf = self.best_clf
		clf.fit(self.train_data, self.train_label)

		predict = clf.predict(self.test_data)
		matrix = confusion_matrix(self.test_label, predict)
		report = classification_report(self.test_label, predict)

		print("confusion matrix:\n")
		print(matrix)
		print("Result:\n")
		print(report)

		self.fd.write("confusion matrix: \n")
		self.fd.write(matrix)
		self.fd.write("\n")

		self.fd.write("Result: \n")
		self.fd.write(report)
		self.fd.write("\n")

		reports = report.strip().split()
		precision = float(reports[-4])
		recall = float(reports[-3])
		f1 = float(reports[-2])

		print("precision, recall, f1: ", precision, recall, f1)
		self.fd.write("precision: %s\n" %precision)
		self.fd.write("recall: %s\n" %recall)
		self.fd.write("F-measure: %s\n" %f1)

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
			exit()

		if self.eval == CV :
			self.crossvalidation()
			self.classification()

		elif self.eval == Bayes :
			self.bayesian()

		else :
			exit()

		self.fd.close()
		exit("done")




if __name__ == "__main__" :
	
	"""
	$1:	njobs
	$2:	config_json	("*.json")
	$3:	traindata	("*.csv" or "*.tsv", headerとindexはなし)
	$4:	testdata	("*.csv" or "*.tsv", headerとindexはなし)
			traindataを分割してtestdataとしたい場合，$4に-1を入力する
	$5:	result_name	(without ext)
	"""

	from sys import argv
	from multiprocessing import cpu_count

	if len(argv) != 6 :
		exit("Error: args [njobs] [config-file (JSON)] [traindata(CSV or TSV)] [testdata(CSV or TSV or -1)] [result name(without ext)]")

	njobs = int(argv[1])
	config_json = argv[2]
	traindata = argv[3]
	testdata = argv[4]
	rslt = argv[5]

	while True :
		if njobs > cpu_count() :
			exit("Error: your chose number of jobs is larger than your PC's number of CPU/.\nYour PC's number of CPU is %d." %cpu_count())
		elif njobs == cpu_count() :
			print("Warning: your chose number of jobs and your PC's number of CPU are same.\nWould you agree that this script continues the processing?")
			key = input("[y / n] >>>")

			if key == "y" :
				break
			if key == "n" :
				print("Do you want to change the number of jobs?")
				key = input("[y / n] >>>")

				if key == "y" :
					print("Input the number of jobs.")
					njobs = int(input(">>>"))
				if key == "n" :
					exit()
				else :
					continue
			else :
				continue
		else :
			break

	clf = Classification(njobs, config_json, traindata, testdata, rslt)
	clf.main()
