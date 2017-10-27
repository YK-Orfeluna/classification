# coding:utf-8

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
target = iris.target
#print(data.shape)
#print(target.shape)


def main(data=iris.data, target=iris.target) :
	print(data.shape)

	feature_names = ["f"+str(i+1) for i in range(data.shape[1])]
	#print(feature_names)

	forest = RandomForestClassifier(
		n_estimators=10000,
		random_state=0,
		n_jobs=6
		)

	forest.fit(data, target)

	importances = forest.feature_importances_.astype(np.str)

	rslt = pd.DataFrame(importances, index=feature_names, columns=["importance"])
	rslt.to_csv("importance.csv")

	pred = forest.predict(data)
	report = classification_report(target, pred)
	print(report)


if __name__ == "__main__" :
	from sys import argv
	from os.path import splitext

	"""
	if len(argv) != 2 :
		exit("Error: args [filename]")
	filename = argv[1]
	"""
	filename = "5_45.csv"
	ext = splitext(filename)[1]

	if ext.find("csv") > 0 :
		dataframe = pd.read_csv(filename, header=None, index_col=None)
	elif ext.find("tsv") > 0 :
		dataframe = pd.read_csv(filename, header=None, index_col=None, delimiter="\t")

	value = dataframe.values
	data = value[:, :-1].copy()
	#data = data[:, :8]
	target = value[:, -1].copy()
	target.astype(np.int64)

	#main()
	main(data, target)

