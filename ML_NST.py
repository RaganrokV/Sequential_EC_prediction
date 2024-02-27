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
data = [arr for arr in SH_data if len(arr) >= 10]
# data = [arr for arr in SZ_data if len(arr) >= 10]
# data = [arr for arr in CN_data if len(arr) >= 10]

result = []

for array in data:
    # 求第1列、第2列、第3列、第4列的和
    sum_col1 = np.sum(array[:, 0])
    avg_col2 = np.mean(array[:, 1])
    sum_col3 = np.sum(array[:, 2])
    sum_col4 = np.sum(array[:, 3])

    # 计算其余列的平均值
    avg_other_cols = np.mean(array[:, 4:], axis=0)

    # 将结果合并为一个1*10的数组
    combined_array = np.array([sum_col1, avg_col2, sum_col3, sum_col4] + list(avg_other_cols))

    # 添加到结果列表中
    result.append(combined_array)

# 将结果列表转换为数组
result_array = np.array(result)
# result_array=data
#%%
"""normalization"""
data_to_normalize = result_array[:, :5]

# 计算最小值和最大值
min_values = data_to_normalize.min(axis=0)
max_values = data_to_normalize.max(axis=0)

# 归一化
normalized_data = (data_to_normalize - min_values) / (max_values - min_values)

# 保存第一列的归一化结果
col1_normalized = normalized_data[:, 0]

# 将归一化后的数据替换回原数组的前6列
result_array[:, :5] = normalized_data
# result_array=result_array/60
#%%
# 检查是否存在 NaN 值并打印包含 NaN 值的数组
for i in range(len(result_array)):
    if np.isnan(np.sum(result_array[i])):
        print(f"第 {i} 个数组中存在 NaN 值：")
        # 将 NaN 值替换为 0
        result_array[i] = np.nan_to_num(result_array[i], 0.0)
#%%
# 计算切分的索引 7:1:2
"""打乱行程顺序"""

random.seed(42)
np.random.seed(42)

# 随机打乱数据

natural_numbers = np.arange(len(result_array))

# 随机打乱数组
np.random.shuffle(natural_numbers)

result_array = [result_array[i] for i in natural_numbers]
train_len = int(0.7 * len(result_array))
val_len = int(0.1 * len(result_array))

# 切分数据集
train_data = np.vstack(result_array[:train_len])
val_data = np.vstack(result_array[train_len:])
test_data=np.vstack(result_array[train_len+val_len:])

#%%
"创建XGBoost模型"
XGB_params = {'learning_rate': 0.1, 'n_estimators': 300,
              'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
              'objective': 'reg:squarederror', 'subsample': 0.8,
              'colsample_bytree': 0.3, 'gamma': 0,
              'reg_alpha': 0.1, 'reg_lambda': 0.1}
XGB = xgb.XGBRegressor(**XGB_params)

# 在当前折叠上进行模型训练和预测
XGB.fit(train_data[:,1:], train_data[:,0])
predictions = XGB.predict(test_data[:,1:])

# 反归一化预测值和真实值
sum_predictions = predictions* (max_values[0] - min_values[0]) + min_values[0]
sum_targets = test_data[:,0]* (max_values[0] - min_values[0]) + min_values[0]

# 计算评估指标并添加到ALL_Metric列表中
metric = np.array(evaluation(sum_targets, sum_predictions))
metric
#%%
"创建SVM模型"
SVR_MODEL = svm.SVR()
SVR_MODEL.fit(train_data[:,1:], train_data[:,0])
predictions = SVR_MODEL.predict(test_data[:,1:])

# 反归一化预测值和真实值
sum_predictions = predictions* (max_values[0] - min_values[0]) + min_values[0]
sum_targets = test_data[:,0]* (max_values[0] - min_values[0]) + min_values[0]

# 计算评估指标并添加到ALL_Metric列表中
metric = np.array(evaluation(sum_targets, sum_predictions))
metric
#%%
"LR"
LR = LinearRegression()
LR.fit(train_data[:,1:], train_data[:,0])
predictions = LR.predict(test_data[:,1:])
# 使用fit函数拟合

sum_predictions = predictions* (max_values[0] - min_values[0]) + min_values[0]
sum_targets = test_data[:,0]* (max_values[0] - min_values[0]) + min_values[0]

# 计算评估指标并添加到ALL_Metric列表中
metric = np.array(evaluation(sum_targets, sum_predictions))
metric
#%%
"MLP"
MLP = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                   batch_size='auto', solver='adam', alpha=1e-04,
                   learning_rate_init=0.001, max_iter=300, beta_1=0.9,
                   beta_2=0.999, epsilon=1e-08)
MLP.fit(train_data[:,1:], train_data[:,0])

predictions=MLP.predict(test_data[:,1:])

sum_predictions = predictions* (max_values[0] - min_values[0]) + min_values[0]
sum_targets = test_data[:,0]* (max_values[0] - min_values[0]) + min_values[0]

# 计算评估指标并添加到ALL_Metric列表中
metric = np.array(evaluation(sum_targets, sum_predictions))
metric