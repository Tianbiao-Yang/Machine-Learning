# =============================================================================
#                           自助法(booststrap)
# =============================================================================


import numpy as np
import pandas as pd
import random

# 生成数据
data = pd.DataFrame(np.random.rand(10,4),columns=list('ABCD'))
data['y'] = [random.choice([0,1]) for i in range(10)]
# print(data)

# 划分数据集
train = data.sample(frac=1.0,replace=True)
test = data.loc[data.index.difference(train.index)].copy()
print(train)
print(test)