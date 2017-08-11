# -*- coding: utf-8 -*-


import json
from os.path import splitext

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, f1_score

def read_data(filename) :
	if splitext(filename)[1] == ".csv" :
		df = pd.read_csv(filename, header=None, index_col=None)
	elif splitext(filename)[1] == ".tsv" :
		df = pd.read_csv(filename, header=None, index_col=None, delimiter="\t")
	else :
		exit("Error: this script can read CSV or TSV file only.")

	data = df.values[:, :-1].copy()		# 最後の1行以外をデータセットにする
	label = df.values[:, -1].copy()		# 最後の1行をラベルにする

	return data, label

class Classification() :
	def __init__(self, njobs, config_json, traindata, testdata, rslt) :

		self.njobs = njobs


		self.param = {}
		self.method = ""
		self.K = 20

		self.load_config(config_json)


		self.train_data = np.array([])
		self.train_label = np.array([])
		self.test_data = np.array([])
		self.test_label = np.array([])
		self.gs_data = np.array([])
		self.gs_label = np.array([])

		self.load_dataset(traindata, testdata)


		self.rslt = rslt


		self.best_clf = None

	def load_config(self, config_json) :
		if splitext(config_json)[1] == ".json" :
			with open(config_json, "r") as fd:
				config = json.load(fd)			# config用のjsonを読み込む
		else :
			exit("Error: this script can read JSON-file as config-file.\nYou should choose JSON-file.")


		self.method = config["method"]		# 分類手法の読み込み

		if self.method != "SVM" and self.method != "KNN" :
			exit("Error: your chose method is not supported by this sciprt.\nYou should choose 'SVM' or 'KNN'.")


		self.K = config["K"]			# K-fold Cross ValidationのKを読み込み

		self.param = config["param"]


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

	def crossvalidation(self) :
		if self.method == "KNN" :
			clf = KNeighborsClassifier()
		elif self.method == "SVM" :
			clf = SVC(probability=True, decision_function_shape="ovr")
			
		gs = GridSearchCV(clf, self.param)
		gs.fit(self.gs_data, self.gs_label)

		gs_result = pd.DataFrame(gs.cv_results_)
		gs_result.to_csv("%s_GS.csv" %self.rslt)

		best_param = gs.best_params_
		print("best parameter: %s" %best_param)

		accuracy = gs.best_score_
		print("best accuracy: %.3f" %accuracy)

		index = gs.best_index_
		accuracy_SD = gs_result["std_test_score"][index]
		print("SD of accuracy: %.3f" %accuracy_SD)

		self.best_clf = gs.best_estimator_
		print(self.best_clf)


	def classification(self) :
		clf = self.best_clf
		clf.fit(self.train_data, self.train_label)

		predict = clf.predict(self.test_data)
		matrix = confusion_matrix(self.test_label, predict)
		report = classification_report(self.test_label, predict)

		print("confusion matrix: ")
		print(matrix)
		print("Result: ")
		print(report)

		reports = report.strip().split()
		precision = float(reports[-4])
		recall = float(reports[-3])
		f1 = float(reports[-2])

		print("precision, recall, f1: ", precision, recall, f1)


	def main(self) :
		self.crossvalidation()
		self.classification()




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

	if len(argv) != 6 :
		exit("Error: missing args")

	njobs = int(argv[1])
	config_json = argv[2]
	traindata = argv[3]
	testdata = argv[4]
	rslt = argv[5]

	clf = Classification(njobs, config_json, traindata, testdata, rslt)
	clf.main()
