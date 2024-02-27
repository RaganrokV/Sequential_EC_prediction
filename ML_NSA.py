# -*- coding: utf-8 -*-
import random
import numpy as np
from My_utils.evaluation_scheme import evaluation
import pickle
import warnings
import xgboost as xgb
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings("ignore")

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
# 创建一个新的列表，用于存储归一化后的数组
data_normalized = []

for arr in data:

    # 提取需要归一化的列数据
    columns_to_normalize = arr[:, [1, 2, 4,5]]

    # 对需要归一化的列按列归一化处理
    normalized_columns = np.divide(columns_to_normalize, np.linalg.norm(columns_to_normalize, axis=0))

    # 将归一化后的列数据替换原始数组中的对应列数据
    arr[:, [1, 2, 4, 5]] = normalized_columns

    # 将修改后的数组添加到新列表中
    data_normalized.append(arr)
#%%
# 检查是否存在 NaN 值并打印包含 NaN 值的数组
for i in range(len(data_normalized)):
    if np.isnan(np.sum(data_normalized[i])):
        print(f"第 {i} 个数组中存在 NaN 值：")
        # 将 NaN 值替换为 0
        data_normalized[i] = np.nan_to_num(data_normalized[i], 0.0)

#%%
"""PADDING DATA"""
max_length = max(len(row) for row in data)
unified_len = ((max_length // 50) + 1) * 50
print(unified_len)
# 定义目标形状
target_shape = (unified_len, 13)

# 新建一个列表，用于存储所有转换后的数组
data_padded = []

for arr in data_normalized:
    # 计算需要填充的行数和列数
    row_pad = target_shape[0] - arr.shape[0]
    col_pad = target_shape[1] - arr.shape[1]

    # 创建一个以零填充的数组
    padded_arr = np.pad(arr, ((0, row_pad), (0, col_pad)), mode='constant', constant_values=0)

    # 将已填充的数组添加到新列表中
    data_padded.append(padded_arr)


"""打乱行程顺序"""

random.seed(42)
np.random.seed(42)

# 随机打乱数据
# 生成包含0到19507的自然数数组
natural_numbers = np.arange(len(data_padded))

# 随机打乱数组
np.random.shuffle(natural_numbers)

data_padded = [data_padded[i] for i in natural_numbers]

num_samples = len(data_padded)

# 创建一个空的三维数组
data_3D = np.empty((num_samples, unified_len, 13))

# 将每个数组复制到相应的位置
for i in range(num_samples):
    arr = data_padded[i]
    data_3D[i] = arr

# 计算切分的索引 7:1:2
train_len = int(0.7 * len(data_3D))
val_len = int(0.1 * len(data_3D))

# 切分数据集
train_data = data_3D[:train_len]
val_data = data_3D[train_len:]
test_data=data_3D[train_len+val_len:]

#%%
"创建XGBoost模型"
XGB_params = {'learning_rate': 0.1, 'n_estimators': 300,
              'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
              'objective': 'reg:squarederror', 'subsample': 0.8,
              'colsample_bytree': 0.3, 'gamma': 0,
              'reg_alpha': 0.1, 'reg_lambda': 0.1}
# XGB_params = {'learning_rate': 0.1, 'n_estimators': 300,
#               'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
#               'objective': 'reg:squarederror', 'subsample': 0.8,
#               'colsample_bytree': 0.3, 'gamma': 0,
#               'reg_alpha': 0.1, 'reg_lambda': 0.1}
XGB = xgb.XGBRegressor(**XGB_params)

# 在当前折叠上进行模型训练和预测
XGB.fit(train_data.reshape(-1,13)[:,1:], train_data.reshape(-1,13)[:,0])
predictions = XGB.predict(test_data.reshape(-1,13)[:,1:])

# 计算评估指标并添加到ALL_Metric列表中
metric = np.array(evaluation(np.sum(predictions.reshape(-1,unified_len),axis=1),
                             np.sum(test_data[:,:,0],axis=1)))
metric
#%%
"创建SVM模型"
SVR_MODEL = svm.SVR()
SVR_MODEL.fit(train_data.reshape(-1,13)[:34470,1:], train_data.reshape(-1,13)[:34470,0])
predictions = SVR_MODEL.predict(test_data.reshape(-1,13)[:,1:])

# 计算评估指标并添加到ALL_Metric列表中
metric = np.array(evaluation(np.sum(predictions.reshape(-1,unified_len),axis=1),
                             np.sum(test_data[:,:,0],axis=1)))
metric
#%%
"LR"
LR = LinearRegression()
LR.fit(train_data.reshape(-1,13)[:,1:], train_data.reshape(-1,13)[:,0])
predictions = LR.predict(test_data.reshape(-1,13)[:,1:])
# 使用fit函数拟合

metric = np.array(evaluation(np.sum(predictions.reshape(-1,unified_len),axis=1),
                             np.sum(test_data[:,:,0],axis=1)))
metric
#%%
"MLP"
MLP = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                   batch_size='auto', solver='adam', alpha=1e-04,
                   learning_rate_init=0.001, max_iter=300, beta_1=0.9,
                   beta_2=0.999, epsilon=1e-08)
MLP.fit(train_data.reshape(-1,13)[:,1:], train_data.reshape(-1,13)[:,0])

predictions=MLP.predict(test_data.reshape(-1,13)[:,1:])

metric = np.array(evaluation(np.sum(predictions.reshape(-1,unified_len),axis=1),
                             np.sum(test_data[:,:,0],axis=1)))
metric