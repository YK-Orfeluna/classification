# -*- coding: utf-8 -*-
#print(__doc__)

##### ライブラリの読み込み
import sys 
import numpy as np
import pylab as pl
from sklearn import neighbors, svm, datasets, grid_search
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

import Tkinter
import tkMessageBox
import tkFileDialog

frame = ["5_", "10_"]
degree = ["30", "45", "60"]
t_name = []

c = np.append(np.array([1e-06, 1e-05, 1e-04, 0.001, 0.01]), np.arange(0.1, 30.1, 0.1))
g = np.append(np.array([0, 1e-06, 1e-05, 1e-04, 0.001, 0.01]), np.arange(0.1, 30.1, 0.1))

for i in frame :
	for j in degree :
		t_name.append(i+j)

for i in t_name :
	print i
	lfilename = i + ".csv"
	tfilename = i + ".csv"
	t = open(i + ".txt", "w")
	"""
	### GUI用のおまじない
	root = Tkinter.Tk()
	root.option_add('*font', ('bFixedSys', 14))
	fTyp=[('csvファイル','*.csv')]
	iDir='.'

	# ファイル選択
	lb=Tkinter.Label(root, text = u'学習データを選択',width=20)
	lb.pack()
	lfilename=tkFileDialog.askopenfilename(filetypes=fTyp,initialdir=iDir)
	lb.configure(text=u'テストデータを選択')
	tfilename=tkFileDialog.askopenfilename(filetypes=fTyp,initialdir=iDir)

	"""
	##### CSVファイルからデータセットを読み込み
	# data1, data2, ....., dataN, label　というCSVファイル

	# 学習用データ読み込み
	data = np.genfromtxt( lfilename, delimiter=',')

	# データファイルの横軸・縦軸要素数
	cols = len(data[0])
	rows = len(data)
	dim = cols-1

	X_learn = data[:,:dim]    
	y_learn = data[:,dim]

	minlabel = int(y_learn.min())
	maxlabel = int(y_learn.max())
	labelN = int(maxlabel-minlabel+1)

	# テスト用データ読み込み
	data = np.genfromtxt( tfilename, delimiter=',')

	# データファイルの横軸・縦軸要素数
	cols = len(data[0])
	rows = len(data)
	t_dim = cols-1

	X_test = data[:,:t_dim]    
	y_test = data[:,t_dim]

	##### 学習データとテストデータの次元数確認
	if dim!=t_dim:
	    tkMessageBox.showerror('showerror','学習データとテストデータの次元数が一致しない')

	    print "dims of learning and test data are not matched (learn:%d test:%d)." % (dim,t_dim)
	    sys.exit(0)

	# 学習データとテストデータを縦に並べる（全データ）
	X_all = np.r_[X_learn, X_test]
	y_all = np.r_[y_learn, y_test]


	##### 認識
	"""
	# k-NNの認識器の準備（グリッドサーチ）
	print("k-nn")
	t.write(str ("k-NN results" + "\n"))
	para_knn = {"n_neighbors":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], "weights":("uniform", "distance")}
	scores = ['accuracy', 'precision', 'recall']
	clf0 = grid_search.GridSearchCV(neighbors.KNeighborsClassifier(), para_knn, cv=20)

	# k-NNの認識器の学習
	clf0.fit(X_learn, y_learn)

	#テスト用データを認識
	y_pred0 = clf0.predict( X_test )
	t.write(str("BestParamater is" + "\n"))
	t.write(str (clf0.best_estimator_))
	t.write(str ("\n"))
	t.write(str ("Each GlidScores are" + "\n"))
	t.write(str (clf0.grid_scores_))
	t.write(str ("\n"))
	"""
	print("svm_liner")
	# SVMの認識器の準備（グリッドサーチ）
	t.write(str("\n" + "SVM" + "\n"))
	para_svm = {"kernel":["linear"], "C":c}
	clf1 = grid_search.GridSearchCV(svm.SVC(), para_svm, cv=20)
	
	# SVMの認識器の学習
	clf1.fit(X_learn, y_learn)

	# テスト用データを認識
	y_pred1 = clf1.predict( X_test )
	t.write(str("BestScore is" + "\n"))
	t.write(str(clf1.best_score_))
	t.write(str ("\n"))
	t.write(str("BestParamater is" + "\n"))
	t.write(str (clf1.best_estimator_))
	t.write(str ("\n"))
	t.write(str ("Each GlidScores are" + "\n"))
	t.write(str (clf1.grid_scores_))
	t.write(str ("\n"))	

	print("svm_rbf")
	t.write(str("\n" + "SVM_rbf" + "\n"))
	para_svm = {"kernel":["rbf"], "C":c, "gamma":g}
	clf1 = grid_search.GridSearchCV(svm.SVC(), para_svm, cv=20)

	# SVMの認識器の学習
	clf1.fit(X_learn, y_learn)

	# テスト用データを認識
	y_pred1 = clf1.predict( X_test )
	t.write(str("BestScore is" + "\n"))
	t.write(str(clf1.best_score_))
	t.write(str ("\n"))
	t.write(str("BestParamater is" + "\n"))
	t.write(str (clf1.best_estimator_))
	t.write(str ("\n"))
	t.write(str ("Each GlidScores are" + "\n"))
	t.write(str (clf1.grid_scores_))


	##### 混同行列（Confusion Matrix）の計算と表示
	#cm0 = confusion_matrix(y_test, y_pred0)

	#print(cm0)

	#target_names = ['class 0', 'class 1']
	#print(classification_report(y_test, y_pred0, target_names=target_names))

	# 交差検定
	# scores = cross_val_score(clf0, X_learn, y_learn, cv=10)
	# print scores.mean()

	# print clf0

	# #cm1 = confusion_matrix(y_test, y_pred1)
	# #print(cm1)

	# #target_names = ['class 0', 'class 1']
	# #print(classification_report(y_test, y_pred1, target_names=target_names))

	# # 交差検定
	# scores = cross_val_score(clf1, X_learn, y_learn, cv=10)
	# print scores.mean()

	# print clf1
	t.close()
print("end")
sys.exit()


##### 散布図行列の表示
lb.configure(text=u'結果・データ表示')

color = ['r','g','b','m','c','y','k']

f, axes = plt.subplots(nrows=dim, ncols=dim) # dim x dim の散布図の準備
for c in range(minlabel,labelN):
    for i in range(dim):
        for j in range(dim):
            axes[i,j].scatter( X_learn[y_learn==c,i], X_learn[y_learn==c,j], color = color[(c-minlabel)%7])
plt.suptitle("plot of learning data")

plt.figure(1)
f, axes = plt.subplots(nrows=dim, ncols=dim) # dim x dim の散布図の準備
for c in range(minlabel,labelN):
    for i in range(dim):
        for j in range(dim):
            axes[i,j].scatter( X_test[y_test==c,i], X_test[y_test==c,j], color = color[(c-minlabel)%7])
plt.suptitle("plot of test data")

plt.show()

