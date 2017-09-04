#coding:utf-8

import numpy as np
from os.path import splitext
from time import ctime

def run(filename, method, outname) :
	# 特徴量ファイルの読み込み
	ext = splitext(filename)[1]
	if ext == ".csv" :
		d = ","
	elif ext == ".tsv" :
		d = "\t"
	else :
		exti("Error: this script supports CSV or TSV only.")

	data = np.genfromtxt(filename, delimiter=d)
	value = data[:, :-1]
	#label = data[:, -1]


	# 特徴量の正規化
	if method == 1 or method == "1" :
		for i in range(value.shape[1]) :
			mean = np.mean(value[:, i])
			sd = np.std(value[:, i])

			value[:, i] = (value[:, i] - mean) / sd

	elif method == 2 or method == "2" :
		for i in range(value.shape[1]) :
			mean = np.mean(value[:, i])
			sd = np.std(value[:, i])

			value[:, i] = (10 * (value[:, i] - mean) / sd + 50) / 100

	else :
		exit("Error: 2nd arg.(method) is 1 or 2 only.")


	# 正規化した特徴量の出力
	data = data.astype(np.str)

	ext = splitext(outname)[1]
	if ext == ".csv" :
		d = ","
	elif ext == ".tsv" :
		d = "\t"

	np.savetxt(outname, data, delimiter=d, fmt="%s")
	print("[%s]\ndone: save file: %s" %(ctime(), outname))


if __name__ == "__main__" :
	from sys import argv

	if len(argv) != 4 :
		exit("Error: missing args\n$1:\tread file name\n$2:\tmethod(1=mean(0) and var(1), 2=to change standard score)\n$3:\tout file name")

	filename = argv[1]
	#filename = "iris.csv"
	method = int(argv[2])
	outname = argv[3]

	run(filename, method, outname)