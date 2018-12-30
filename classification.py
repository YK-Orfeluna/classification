# -*- coding: utf-8 -*-

import tkinter
from tkinter import Entry, Label, Button, filedialog, END, \
					Radiobutton, IntVar, messagebox, Checkbutton, BooleanVar, \
					CENTER
from multiprocessing import cpu_count

import json
from os.path import splitext, exists
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
from sklearn.ensemble import RandomForestClassifier

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
RF = "RF"

CV = "CV"
Bayes = "Bayes"
auto = "auto"

def k(n) :
	out =  (1 + np.log(n) / np.log(2)) * 4
	return int(round(out, 0))

def read_data(filename, flagH) :					# ファイル読み込み（csv/tsvに限る）
	if splitext(filename)[1]==".csv" :
		delimiter = ","
	elif splitext(filename)[1]==".tsv" :
		delimiter = "\t"
	else :
		messagebox.showwarning("WARNING", "Your chosed train/test data is not CSV/TSV-file.")
		return 0

	if flagH :
		H = 0
	else :
		H = None

	try :
		df = pd.read_csv(filename, header=H, delimiter=delimiter)
	except OSError :
		df = pd.read_csv(filename, header=H, delimiter=delimiter, engine="python")
	except FileNotFoundError :
		messagebox.showwarning("WARNING", "Your chosed train/test data is not found.")
		return 0

	data = df.values[:, :-1]			# 最後の1行以外をデータセットにする
	label = df.values[:, -1]			# 最後の1行をラベルにする

	return data, label

class Classification() :
	def __init__(self, njobs, config_json, traindata, testdata, outdir) :

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

		self.load_config(config_json)


		if outdir[-1] == "/" :
			self.outdir = outdir[: -1]
		else :
			self.outdir = outdir

		if not exists(self.outdir) :
			getoutput("mkdir %s" %self.outdir)
			print("mkdir %s" %self.outdir)


		self.best_clf = None

		self.fd = open("%s/rslt.txt" %outdir, "w")

	def load_config(self, config_json) :			# 設定用のjsonファイルを読み込む
		try :
			with open(config_json, "r") as fd :
				config = json.load(fd)
		except FileNotFoundError :
			messagebox.showwarning("WARNING", "Your chosed json-file is not found.")
			return 0

		method = config["method"]
		if method == SVM or method == KNN or method == Kmeans or method == GMM or method == RF:
			self.method = method		# 分類手法の読み込み
		else :
			supported = [SVM, KNN, Kmeans, GMM, RF]
			messagebox.showwarning("WARNING", "Your chosed method (%s) is not supported on this script.\nSupported: %s" %(method, supported))
			return 0

		cv = config["K"]
		if cv == auto :					# "auto"の場合，サンプル数から自動的にCrossValidationの回数を決める
			self.cv = k(self.gs_data.shape[0])
		else :
			try :
				self.cv = int(cv)
			except ValueError :
				messagebox.showwarning("WARNING", "Your have to fill in \"K in JSON\" as integer or \"auto\"")
				return 0

		evaluation = config["evaluation"]
		if evaluation == CV:
			self.eval = evaluation
		else :
			messagebox.showwarning("WARNING", "your chosed CV-method (%s) is not supported on this script.You have to choose %s" %(evaluation, CV))
			return 0

		self.param = config["param"]

		print("[%s]done: read %s" %(ctime(), config_json))


	def load_dataset(self, traindata, testdata) :
		if testdata == -1 or testdata == "-1" :								# testdataが-1だった場合，traindataを分割する
			data, label = read_data(traindata, trainH.get())

			self.gs_data = data.copy()
			self.gs_label = label.copy()

			self.train_data = data[0::2, :].copy()
			self.train_label = label[0::2].copy()

			self.test_data = data[1::2, :].copy()
			self.test_label = label[1::2].copy()

			print("[%s]\ndone: read %s, %s" %(ctime(), traindata, testdata))

		else :
			self.train_data, self.train_label = read_data(traindata, trainH.get())		# 学習データの読み込み
			self.test_data, self.test_label = read_data(testdata, testH.get())			# テストデータの読み込み

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

	def main(self) :
		if self.method == KNN :
			self.clf = KNeighborsClassifier()
		elif self.method == SVM :
			self.clf = SVC(probability=True, decision_function_shape="ovr")
		elif self.method == Kmeans :
			self.clf = KMeans()
		elif self.method == GMM :
			self.clf = GaussianMixture()
		elif self.method == RF :
			self.clf = RandomForestClassifier()
		else :
			exit()

		if self.eval == CV :
			self.crossvalidation()
			self.classification()

		else :
			exit()

		self.fd.close()

def readData(filename, H=None) :
	ext = splitext(filename)[1]
	if ext==".csv" :
		delimiter=","
	elif ext==".tsv" :
		delimiter="\t"

	try :
		df = pd.read_csv(filename, header=H, delimiter=delimiter,)
	except OSError :
		df = pd.read_csv(filename, header=H, delimiter=delimiter, engine="python")
	return df

def _getJSON() :
	json_name = filedialog.askopenfilename(title="Choose JSON-file", initialdir=".", filetypes=[("JSON", "*.json")])
	if json_name!="":
		entryJSON.delete(0, END)
		entryJSON.insert(END, json_name)

def _getTrain() :
	train_name = filedialog.askopenfilename(title="Choose CSV/TSV-file", initialdir=".", filetypes=[("CSV", "*.csv"), ("TSV", "*.tsv")])
	if train_name!="":
		entryTrain.delete(0, END)
		entryTrain.insert(END, train_name)

def _getTest() :
	test_name = filedialog.askopenfilename(title="Choose CSV/TSV-file", initialdir=".", filetypes=[("CSV", "*.csv"), ("TSV", "*.tsv")])
	if test_name!="":
		entryTest.delete(0, END)
		entryTest.insert(END, test_name)

def _normalization() :
	filename = entryTrain.get()
	caseN = normV.get()
	flagH = trainH.get()

	if flagH:
		H = 0
	else :
		H = None

	if splitext(filename)[1]==".csv" :
		delimiter = ","
	elif splitext(filename)[1]==".tsv" :
		delimiter = "\t"
	else :
		messagebox.showwarning("WARNING", "Your chosed file is not CSV/TSV-file.")
		return 0

	try :
		df = pd.read_csv(filename, header=H, delimiter=delimiter)
	except OSError :
		df = pd.read_csv(filename, header=H, delimiter=delimiter, engine="python")
	except FileNotFoundError :
		messagebox.showwarning("WARNING", "Your chosed CSV/TSV-file is not found.")

	data = df.values

	X = data[:, :-1]
	y = np.array([data[:, -1]]).T

	if caseN==0 :
		mean = np.mean(X, axis=0)
		std = np.std(X, axis=0, ddof=1)
		normalized = (X-mean) / std
	if caseN==1 :
		min_ = np.amin(X, axis=0)
		max_ = np.amax(X, axis=0)
		normalized = (X-min_) / (max_-min_) 
	
	outdata = np.append(normalized, y, axis=1)

	if flagH :
		header = np.array([df.columns.values])
		outdata = np.append(header, outdata, axis=0)

	ext = splitext(filename)[1]
	outname = splitext(filename)[0] + "_norm" + ext
	if ext==".csv" :
		delimiter = ","
	elif ext==".tsv" :
		delimiter = "\t"
	np.savetxt(outname, outdata, fmt="%s", delimiter=delimiter)

	messagebox.showinfo("INFORMATIONS", "%s\n is converted to\n%s" %(filename, outname))
	entryTrain.delete(0, END)
	entryTrain.insert(END, outname)

def _main() :
	try :
		njobs = int(entryJob.get())
		if njobs>Ncpu :
			messagebox.showwarning("WARNING", "Your choosed \"Numnber of jobs\" is larger than this PC\'s \"Number of CPU\"")
			entryJob.delete(0, END)
			entryJob.insert(END, "1")
			return 0
	except ValueError :
		messagebox.showwarning("WARNING", "You have to fill out \"Number of jobs\" as integer.")
		return 0

	jsonname = entryJSON.get()
	if jsonname=="" :
		messagebox.showwarning("WARNING", "You have not choose json-file yet.")
		return 0

	trainname = entryTrain.get()
	if trainname=="" :
		messagebox.showwarning("WARNING", "You have not choose train data yet.")
		return 0

	testname = entryTest.get()
	if testname=="" :
		if splitV.get() :
			testname = -1
		else :
			messagebox.showwarning("WARNING", "You have not choose test data or check \"Split Train Data\"")
			return 0

	rslt = entryRslt.get()
	if rslt=="" :
		messagebox.showwarning("WARNING", "You have not fill out \"Result\'s Name\"")
		return 0

	clf = Classification(njobs, jsonname, trainname, testname, rslt)
	clf.main()

def _quit() :
	root.destroy()
	exit()

if __name__ == "__main__" :
	root = tkinter.Tk()
	root.title("Classification")
	root.geometry("500x500")

	"""JSONファイルを選択する"""
	labelJSON = Label(root, text="JSON-file\'s name")
	labelJSON.place(x=10, y=40)

	entryJSON = Entry(root, width=20)
	entryJSON.place(x=150, y=40)

	buttonJSON = Button(root, text="Choose JSON-file", command=_getJSON)
	buttonJSON.place(x=300, y=40)

	"""学習データを選択する"""
	labelTrain = Label(root, text="Train Data (CSV/TSV)")
	labelTrain.place(x=10, y=70)

	entryTrain = Entry(root, width=20)
	entryTrain.place(x=150, y=70)

	buttonTrain = Button(root, text="Choose Train Data (CSV/TSV)", command=_getTrain)
	buttonTrain.place(x=300, y=70)

	trainH = BooleanVar()
	trainH.set(False)

	checkTrainH = Checkbutton(root, text="Header of Train Data")
	checkTrainH.place(x=150, y=100)

	"""学習データの正規化"""
	normV = IntVar()
	normV.set(0)

	radioZ = Radiobutton(root, text="z-score", value=0, variable=normV)
	radioZ.place(x=150, y=130)

	radioM = Radiobutton(root, text="Min-Max", value=1, variable=normV)
	radioM.place(x=250, y=130)

	buttonNorm = Button(root, text="NORMALIZE", command=_normalization)
	buttonNorm.place(x=350, y=130)

	"""評価データを取得"""
	labelTest = Label(root, text="Test Data (CSV/TSV)")
	labelTest.place(x=10, y=160)

	entryTest = Entry(root, width=20)
	entryTest.place(x=150, y=160)

	buttonTest = Button(root, text="Choose Test Data (CSV/TSV", command=_getTest)
	buttonTest.place(x=300, y=160)

	"""学習データを分割して評価データにする"""
	splitV = BooleanVar()
	splitV.set(False)

	checkTest = Checkbutton(root, text="Split Train Data to Train/Test Data", variable=splitV)
	checkTest.place(x=150, y=190)

	labelSplit = Label(root, text="Ratio of Test Data (0.0-1.0)")
	labelSplit.place(x=150, y=210)

	entrySplit = Entry(root, width=5)
	entrySplit.place(x=350, y=210)
	entrySplit.insert(END, "0.3")

	"""処理スレッド数を設定する"""
	labelJob = Label(root, text="Number of Jobs")
	labelJob.place(x=10, y=250)

	entryJob = Entry(root, width=5)
	entryJob.place(x=150, y=250)
	entryJob.insert(END, "1")

	Ncpu = cpu_count()
	labelCPU = Label(root, text="Number of CPUs is \"%d\"" %Ncpu)
	labelCPU.place(x=200, y=250)

	"""結果名を設定する"""
	labelRslt = Label(root, text="Result\'s Name\n(without extention)")
	labelRslt.place(x=10, y=280)

	entryRslt = Entry(root, width=20)
	entryRslt.place(x=150, y=280)


	"""処理開始"""
	buttonMain = Button(root, text="Start Machine Learning", command=_main)
	buttonMain.place(relx=0.5, y=350, anchor=CENTER)

	buttonQuit = Button(root, text="EXIT", command=_quit)
	buttonQuit.place(relx=0.5, y=400, anchor=CENTER)

	root.mainloop()

	exit()