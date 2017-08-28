# classification.py

## GridSearchによる機械学習のチューニング等

### 引数
* $1: njobs
* $2: 設定用のjsonファイル
* $3: 教師データ（CSV, TSVに限る）
* $4: 学習データ（なければ，-1を指定することで，教師データを分割）
* $5: 出力ディレクトリ（なければ，自動生成）

#### 教師・評価データについて
* 教師データは，iris.csvを参考にする
	+ 最終列が教師ラベル，それ以外は特徴量
* 学習データは，教師データ同様の形式で
	+ ただし，教師ラベルはなしで

### 出力用のjsonファイルについて
* "method": 分類手法
	+ 現在対応しているもの（KNN, SVM, KMeans, GMM）
* "evaluation": 評価手法
	+ 現在対応しているもの（CV）
* "K": Cross Validationの回数
	+ 任意の整数値 or auto
	+ ただし，数が大きすぎると，sklearn内部のエラーが発生する可能性がある
* "param": グリッドサーチのパラメータ群
	+ 詳しくは[公式リファレンス]( http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)を参照
