# Cross valicdation
## Reason
由于我们是将数据集分成S和T分别用于训练和测试，但有时我们面临着这样的问题，测试集的信息足以颠覆已训练好的模型，亦是过拟合的情况。为解决此类问题，我们应该准备一部分数据集-验证集，使模型训练完成后，对模型进行评估，最后再在测试集上进行评估，应用交叉验证策略（cv）进行解决。
## Defination
交叉验证法：先把数据集分成k个大小相似的互斥子集，在通过分层抽样以保证数据分布一致，然后用k-1个子集的并集作为训练集，余下的子集作为测试集，这样可以获得k组训练/测试集合，进行k次训练和测试，返回k个结果的均值。
## Thinking
过拟合情况 --> cv的指标(scores) --> 交叉验证迭代器 --> 应用
## 计算交叉验证的指标(scores)
* k-折交叉相关：cross_val_score(clf, iris.data, iris.target, cv=k)
* 改变scoring的计算方式：scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring = 'f1_macro')
	* scoring 参数: 定义模型评估规则(有[回归，分类，聚类](http://sklearn.apachecn.org/cn/0.19.0/modules/model_evaluation.html#scoring-parameter)之分)
* 其他交叉验证策略：
	* cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
	* cross_val_score(clf, iris.data, iris.target, cv = cv)
* 数据预处理(标准化,均值去除和按方差比例缩放):
	* scaler = preprocessing.StandardScaler().fit(X_train)
	* X_train_transformed = scaler.transform(X_train)
	* X_test_transformed = scaler.transform(X_test)
* cross_validate 函数和多度量评估:
	* scoring = ['precision_macro', 'recall_macro']
	* scoring = {'prec_macro': 'precision_macro', 'rec_micro': make_scorer(recall_score, average='macro')}
* 通过交叉验证获取预测:
	* predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)
	* metrics.accuracy_score(iris.target, predicted) 
## 交叉验证迭代器
针对不同的数据类型，我们选用不同的交叉验证迭代器进行处理，主要包括以下几个方面：
   1. 针对IID类型数据：
 	* k折、重复 K-折交叉验证、留一交叉验证(LOO)、留P交叉验证(LPO)、随机排列交叉验证
   2. 在目标类别的分布上可能表现出很大的不平衡性：例如，可能会出现比正样本多数倍的负样本：
       * 分层k折、 分层随机 Split
   3. 样本的分布依赖于样本groups的数据：例如从多个患者收集医学数据，从每个患者身上采集多个样本，这样的数据很可能取决于个人群体
       # 组k-flod、留一组交叉验证、留p组交叉验证、Group Shuffle Split



























