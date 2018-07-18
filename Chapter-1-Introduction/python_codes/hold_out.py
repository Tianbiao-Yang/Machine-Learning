# =============================================================================
#                               Hold out
# =============================================================================

from sklearn.model_selection import train_test_split
import numpy as np

# X = np.random.rand(10,4)
X = np.array([[0,1],[2,3],[4,5],[6,7],[8,9]])
y = [1,1,0,0,1] 
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
# 分层采样，当random_state不为o或不填时，为分层采样
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=1,stratify=y)



