# =============================================================================
#                                  cross validation                              
# =============================================================================

'''
   由于我们是将数据集分成S和T分别用于训练和测试，但有时我们面临着这样的问题，测试集的信息足
以颠覆已训练好的模型，亦是过拟合的情况。为解决此类问题，我们应该准备一部分数据集-验证集，使
模型训练完成后，对模型进行评估，最后再在测试集上进行评估，应用交叉验证策略（cv）进行解决。
    本codes从“过拟合情况 --> cv的指标(scores) --> 交叉验证迭代器 --> 应用”进行讲述
'''

# --* 快速采样，但出现过拟合情况 *-- #
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

# 载入iris数据，并划分数据集
iris = datasets.load_iris()
# (iris.data.shape,iris.target.shape) # 观察维数
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=0)

# 训练模型，并测试
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)
# print("Accuracy: %0.2f " % clf.score(X_test, y_test))


# ====================================================
#                计算交叉验证的指标(scores)
# ====================================================

# --* 应用k-折交叉相关，减缓过拟合情况 *-- #
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
# X:features  y:targets  cv:k, 5 flod
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
(scores.mean(), scores.std() * 2)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# 改变scoring的计算方式, scoring 参数: 定义模型评估规则(有回归，分类，聚类之分)
scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring = 'f1_macro')
(scores.mean(), scores.std() * 2)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# --* 使用其他交叉验证方法表示 *--
from sklearn.model_selection import ShuffleSplit

n_samples = iris.data.shape[0]
# ?? ShuffleSplit啥意思，传入一个交叉验证迭代器来使用其他交叉验证策略
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
cross_val_score(clf, iris.data, iris.target, cv = cv)
# print(cross_val_score(clf, iris.data, iris.target, cv = cv))


# --* 对样本空间进行改造，加入predictor（预测器），从训练集中学习预测，预处理 *--
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

# 数据预处理，(标准化,均值去除和按方差比例缩放)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)
# 训练模型并打分
clf = svm.SVC(C=1).fit(X_train_transformed,y_train)
clf.score(X_test_transformed,y_test)
# print("Accuracy: %0.2f" % clf.score(X_test_transformed,y_test))
# 合并评估器，来训练模型
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
cross_val_score(clf, iris.data, iris.target, cv = cv)
# print(cross_val_score(clf, iris.data, iris.target, cv = cv))


# --* cross_validate 函数和多度量评估 *--
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics.scorer import make_scorer

scoring = ['precision_macro', 'recall_macro']
clf = svm.SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
                        cv=5, return_train_score=False)
# sorted(scores.keys())
scores['test_recall_macro']       
# 自定义得分函数
scoring = {'prec_macro': 'precision_macro',
           'rec_micro': make_scorer(recall_score, average='macro')}
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
                        cv=5, return_train_score=True)
# sorted(scores.keys())                 
scores['train_rec_micro']     
# 使用单一指标
scores = cross_validate(clf, iris.data, iris.target,
                        scoring='precision_macro')
# sorted(scores.keys())
scores['test_score']


# --* 通过交叉验证获取预测 *--
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)
metrics.accuracy_score(iris.target, predicted) 



# ====================================================
#                    交叉验证迭代器
# ====================================================
'''
   针对不同的数据类型，我们选用不同的交叉验证迭代器进行处理，主要包括以下几个方面：
   1. 针对IID类型数据：
       k折、重复 K-折交叉验证、留一交叉验证(LOO)、留P交叉验证(LPO)、随机排列交叉验证
   2. 在目标类别的分布上可能表现出很大的不平衡性：
       例如，可能会出现比正样本多数倍的负样本：
       分层k折、 分层随机 Split
   3. 样本的分布依赖于样本groups的数据：
       例如从多个患者收集医学数据，从每个患者身上采集多个样本，这样的数据很可能取决于个人群体
       组k-flod、留一组交叉验证、留p组交叉验证、Group Shuffle Split
'''


# --* 针对IID分布数据 *--

# k折 (由n_splits确定)
import numpy as np
from sklearn.model_selection import KFold
X = np.array([[0., 0.], [1., 1.], [-1., -1.], [2., 2.]])
y = np.array([0, 1, 0, 1])
kf = KFold(n_splits=2)
for train, test in kf.split(X):
    print("%s %s" % (train,test))
X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

# 重复k折交叉验证
from sklearn.model_selection import RepeatedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2]])
random_state = 1
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
for train, test in rkf.split(X):
    print("%s %s" % (train, test))
# X_train, X_test = X[train], X[test]

# 留一交叉验证(LOO)
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
for train, test in loo.split(X):
    print("%s %s" % (train, test))

# 留p交叉验证（LPO）
from sklearn.model_selection import LeavePOut
lpo = LeavePOut(2)
for train, test in lpo.split(X):
    print("%s %s" % (train, test))

# 随机排列交叉验证
from sklearn.model_selection import ShuffleSplit
# X = np.arange(5)
ss = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
for train_index, test_index in ss.split(X):
    print("%s %s" % (train_index, test_index))
    # X[test_index], X[train_index]


# --* 基于类标签、具有分层的交叉验证迭代器 *-- 
# 分层k折
from sklearn.model_selection import StratifiedKFold
X = np.array(np.ones(10))
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
skf = StratifiedKFold(n_splits=3)
for train_index, test_index in skf.split(X, y):
    print("%s %s" % (train_index, test_index))
    # X_train, X_test = X[train_index], X[test_index]
    # y_train, y_test = y[train_index], y[test_index]
    # print(X_train, X_test, y_train, y_test)

# 分层随机 Split
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
for train_index, test_index in sss.split(X, y):
    print("%s %s" % (train, test))
    # X_train, X_test = X[train_index], X[test_index]
    # y_train, y_test = y[train_index], y[test_index]
    # print(X_train, X_test, y_train, y_test)


# --* 分组数据类型的数据的交叉验证迭代器 *--
# 组 k-flod
from sklearn.model_selection import GroupKFold
# 科目成绩数据
X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
gkf = GroupKFold(n_splits=3)
for train, test, in gkf.split(X, y, groups= groups):
    print("%s %s" % (train, test))

# 留一组交叉验证
from sklearn.model_selection import LeaveOneGroupOut
# 在多个实验的情况下
X = [1, 5, 10, 50, 60, 70, 80]
y = [0, 1, 1, 2, 2, 2, 2]
groups = [1, 1, 2, 2, 3, 3, 3]
logo = LeaveOneGroupOut()
for train, test in logo.split(X, y, groups=groups):
    print("%s %s" % (train, test))

# 留p组交叉验证
from sklearn.model_selection import LeavePGroupsOut
lpgo = LeavePGroupsOut(n_groups=2)
for train, test in lpgo.split(X, y, groups=groups):
    print("%s %s" % (train, test))

# Group Shuffle Split 生成随机划分分区的序列
from sklearn.model_selection import GroupShuffleSplit
X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 0.001]
y = ["a", "b", "b", "b", "c", "c", "c", "a"]
groups = [1, 1, 2, 2, 3, 3, 4, 4]
gss = GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
for train, test in gss.split(X, y, groups=groups):
    print("%s %s" % (train, test))   



# ====================================================
#               交叉验证在时间序列数据中应用
# ====================================================
'''
   时间序列数据是指在不同时间点上收集到的数据，
   这类数据反映了某一事物、现象等随时间的变化状态或程度
'''
from sklearn.model_selection import TimeSeriesSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit(n_splits=3)
for train, test in tscv.split(X):
    print("%s %s" % (train, test))



# =============================================================================
#                                     END
# =============================================================================