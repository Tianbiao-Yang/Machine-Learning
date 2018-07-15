# =============================================================================
#                                   cross validation                              
# =============================================================================


# --* 快速采样，但出现过拟合情况 *--
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
print("Accuracy: %0.2f " % clf.score(X_test, y_test))


# --* 应用k-折交叉相关，减缓过拟合情况 *--
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
# X:features  y:targets  cv:k, 5 flod
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# 改变scoring的计算方式, scoring 参数: 定义模型评估规则(有回归，分类，聚类之分)
# scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring = 'f1_macro')
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# 
from sklearn.model_selection import ShuffleSplit
n_samples = iris.data.shape[0]
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
print(cv)
cross_val_score(clf, iris.data, iris.target, cv = cv)




