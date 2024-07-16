# -*- coding: utf-8 -*-
import random
import numpy as np
from My_utils.evaluation_scheme import evaluation
import pickle
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
#%%

# 从文件中读取数据
with open('8-EC-speed/route_data/SH_route_new.pkl', 'rb') as f:
    SH_data = pickle.load(f)

with open('8-EC-speed/route_data/SZ_route_new.pkl', 'rb') as f:
    SZ_data = pickle.load(f)

with open('8-EC-speed/route_data/CN_route_new.pkl', 'rb') as f:
    CN_data = pickle.load(f)



#%%
"""normalization"""
# data = [arr for arr in SH_data if len(arr) >= 10]
# data = [arr for arr in SZ_data if len(arr) >= 10]
data = [arr for arr in CN_data if len(arr) >= 10]


# 检查是否存在 NaN 值并打印包含 NaN 值的数组
for i in range(len(data)):
    if np.isnan(np.sum(data[i])):
        print(f"第 {i} have NaN 值：")
        # 将 NaN 值替换为 0
        data[i] = np.nan_to_num(data[i], 0.0)


#%%

"""打乱行程顺序"""

random.seed(42)
np.random.seed(42)

# 随机打乱数据
natural_numbers = np.arange(len(data))

# 随机打乱数组
np.random.shuffle(natural_numbers)

data_padded = [data[i] for i in natural_numbers]
#%%

# 计算切分的索引 7:1:2
train_len = int(0.7 * len(data_padded ))
val_len = int(0.1 * len(data_padded ))

# 切分数据集
train_data = data_padded [:train_len]
val_data = data_padded [train_len:]
test_data=data_padded [train_len+val_len:]
#%%
train_ = []

for array in train_data:
    # 提取第一列和第四列
    col0 = array[:, 0]  # 第一列
    col1 = array[:, 1]  # 第一列
    col3 = array[:, 3]  # 第四列
    col4 = array[:, 4]  # 第四列

    # 计算求和
    sum_col0 = np.sum(col0)
    mean_col1 = np.mean(col1)
    sum_col3 = np.sum(col3)
    mean_col4 = np.mean(col4)

    # 将结果存储到sums列表中
    train_.append(np.array([sum_col0 , mean_col1, sum_col3, mean_col4]))

train=np.vstack(train_)

X = train[:, 1:]  # 特征变量
y = train[:, 0]   # 目标变量

# 创建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X, y)
#%%

test_ = []

for array in test_data:
    # 提取第一列和第四列
    col0 = array[:, 0]  # 第一列
    col1 = array[:, 1]  # 第一列
    col3 = array[:, 3]  # 第四列
    col4 = array[:, 4]  # 第四列

    # 计算求和
    sum_col0 = np.sum(col0)
    mean_col1 = np.mean(col1)
    sum_col3 = np.sum(col3)
    mean_col4 = np.mean(col4)

    # 将结果存储到sums列表中
    test_.append(np.array([sum_col0 , mean_col1, sum_col3, mean_col4]))

test=np.vstack(test_)


#%%

predictions = model.predict(test[:,1:])

targets=test[:,0]

"""method1"""
Metric=np.array(evaluation(targets.reshape(-1), predictions.reshape(-1)))
print("acc:",Metric)




