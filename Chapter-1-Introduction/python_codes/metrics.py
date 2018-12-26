# ==================================================================================================
#                                           性能度量metrics
# ==================================================================================================

from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

# 对于无额外参数的，直接可以用，否则需要make_scorer转换后，再调用
ftwo_scorer = make_scorer(fbeta_score,beta=2)
gird = GridSearchCV(LinearSVC(),param_grid={'C':[1, 10]}, scoring=ftwo_scorer)


# --* 分类任务指标 *--
# matrics里的一些指标基本是为二分类定义的，如f1_score, roc_auc_score
# 默认情况下，仅评估正标签

# 精度与错误率
import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score(y_pred, y_true)
accuracy_score(y_pred, y_true, normalize=False)
accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))

# Cohen’s kappa系数
from sklearn.metrics import cohen_kappa_score
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
cohen_kappa_score(y_true, y_pred)

# 混淆矩阵
from sklearn.metrics import confusion_matrix
y_true = [0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# 分类报告
from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 1, 0]
target_names = ['1', '2', '3']
print(classification_report(y_true, y_pred, target_names=target_names))

# 汉明损失 hamming loss
from sklearn.metrics import hamming_loss
y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]
hamming_loss(y_true, y_pred)
hamming_loss(np.array([[0, 1], [1, 1]]), np.zeros((2, 2)))

# Jaccard 相似系数 
from sklearn.metrics import jaccard_similarity_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
jaccard_similarity_score(y_true, y_pred)
jaccard_similarity_score(y_true, y_pred, normalize=False)
jaccard_similarity_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))

# 准确率、召回率和 F-measures
# 二分类
from sklearn import metrics
y_pred = [0, 1, 0, 0]
y_true = [0, 1, 0, 1]
metrics.precision_score(y_true, y_pred)
metrics.recall_score(y_true, y_pred)
metrics.f1_score(y_true, y_pred)
# beta=1时，为标准的F1
metrics.fbeta_score(y_true, y_pred, beta=1)
# beta<1时，对准确率有更大的影响
metrics.fbeta_score(y_true, y_pred, beta=0.5)
# beta>1时，对召回率有更大的影响
metrics.fbeta_score(y_true, y_pred, beta=2)
# question：不太懂support是什么意思
metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5)

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
precision, recall, threshold = precision_recall_curve(y_true, y_scores)
average_precision_score(y_true, y_scores)

# 多分类
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
metrics.precision_score(y_true, y_pred, average='macro')  
metrics.recall_score(y_true, y_pred, average='micro')
metrics.f1_score(y_true, y_pred, average='weighted')  
metrics.fbeta_score(y_true, y_pred, average='macro', beta=0.5)  
metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5, average=None)
metrics.recall_score(y_true, y_pred, labels=[2], average='micro')
metrics.precision_score(y_true, y_pred, labels=[0, 1, 2, 3], average='macro')

# 准确率、召回率和F1



