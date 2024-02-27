# -*- coding: utf-8 -*-
import random
import torch.nn.init as init
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from My_utils.evaluation_scheme import evaluation
import pickle
import torch.utils.data as Data
import warnings
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

    # """test the performance of speed"""
    # arr[:, 1] = 0

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
trainX=torch.tensor(train_data[:,:,1:]).float()
trainY=torch.tensor(train_data[:,:,0]).unsqueeze(2).float()
vaildX=torch.tensor(val_data[:,:,1:]).float()
vaildY=torch.tensor(val_data[:,:,0]).unsqueeze(2).float()

train_dataset = Data.TensorDataset(trainX, trainY)
dataloaders_train = Data.DataLoader(dataset=train_dataset,
                                batch_size=16, shuffle=False,
                                generator=torch.Generator().manual_seed(42))

vaild_dataset = Data.TensorDataset(vaildX, vaildY)
dataloaders_valid= Data.DataLoader(dataset=vaild_dataset,
                                    batch_size=16,
                                   shuffle=False,
                                   generator=torch.Generator().manual_seed(42))

#%%
class LSTM(nn.Module):
    def __init__(self, seq_len, hidden_size, num_layers, pre_len):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.1,
            batch_first=True)
        self.out = nn.Linear(hidden_size*2, pre_len)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param.data)  # 使用均匀分布进行权重初始化
            elif 'bias' in name:
                init.constant_(param.data, 0.0)  # 将偏置初始化为0

    def forward(self, x):
        temp, _ = self.lstm(x)
        s, b, h = temp.size()
        temp = temp.reshape(s * b, h)
        outs = self.out(temp)
        lstm_out = outs.reshape(s, b, -1)

        return lstm_out

# %%
def train_model(model, dataloaders_train, dataloaders_valid, epochs, optimizer, criterion, batch_size):
    train_loss_all = []
    val_loss_all = []
    best_val_loss = float("inf")
    best_model = None
    model.train()  # Turn on the train mode
    total_loss = 0.

    for epoch in range(epochs):
        train_loss = 0
        train_num = 0
        for step, (x, y) in enumerate(dataloaders_train):

            optimizer.zero_grad()

            x, y = x.to(device), y.to(device)

            pre_y = model(x)

            loss = criterion(pre_y, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # 梯度裁剪，放backward和step直接
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_num += x.size(0)

            total_loss += loss.item()
            log_interval = int(len(dataloaders_train.dataset) / batch_size / 5)
            if (step + 1) % log_interval == 0 and (step + 1) > 0:
                cur_loss = total_loss / log_interval
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.6f} | '
                      'loss {:5.5f}'.format(
                    epoch, (step + 1), len(dataloaders_train.dataset) // batch_size, optimizer.param_groups[0]['lr'],
                    cur_loss))
                total_loss = 0

        if (epoch + 1) % 5 == 0:
            print('-' * 89)
            print('end of epoch: {}, Loss:{:.7f}'.format(epoch + 1, loss.item()))
            print('-' * 89)

        train_loss_all.append(train_loss / train_num)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(dataloaders_valid):
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()

        val_loss /= len(dataloaders_valid.dataset)
        val_loss_all.append(val_loss)

        scheduler.step(val_loss)

        model.train()  # 将模型设置回train()模式

        print('Epoch: {} Validation Loss: {:.6f}'.format(epoch + 1, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            print('best_Epoch: {} Validation Loss: {:.6f}'.format(epoch + 1, best_val_loss))

    # 绘制loss曲线
    plt.figure()
    adj_val_loss = [num * (train_loss_all[0]/val_loss_all[0]-1) for num in val_loss_all]
    plt.plot(range(1, epochs + 1), train_loss_all, label='Train Loss')
    plt.plot(range(1, epochs + 1), adj_val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return best_model
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

input_size = 12  # 输入大小
hidden_size = 300  # 隐藏层大小
num_layers = 2  # 层数
batch_size=16
epochs=300
output_size = 1  # 输出大小
lr=0.0001

LSTM_model = LSTM(input_size, hidden_size, num_layers,
                       output_size).to(device)
optimizer = torch.optim.AdamW(LSTM_model.parameters(),  lr=lr,
                              betas=(0.9, 0.9999), weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       patience=5, factor=0.90)

criterion = nn.MSELoss()

best_model= train_model(LSTM_model, dataloaders_train, dataloaders_valid,
                                        epochs, optimizer, criterion,
                        batch_size)
#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

input_size = 12  # 输入大小
hidden_size = 300  # 隐藏层大小
num_layers = 2  # 层数
output_size = 1  # 输出大小

LSTM_model = LSTM(input_size, hidden_size, num_layers, output_size)

total_params = count_parameters(LSTM_model)
print(f"Total Trainable Parameters: {total_params}")
#%%
# torch.save(best_model, '8-EC-speed/model/LSTM_SH_SA.pth')
# torch.save(best_model, '8-EC-speed/model/LSTM_SH_SANV.pth')
# torch.save(best_model, '8-EC-speed/model/LSTM_SZ_SA.pth')
# torch.save(best_model, '8-EC-speed/model/LSTM_SZ_SANV.pth')
# torch.save(best_model, '8-EC-speed/model/LSTM_CN_SA.pth')
# torch.save(best_model, '8-EC-speed/model/LSTM_CN_SANV.pth')
#%%
"predict"
testX=torch.tensor(test_data[:,:,1:]).float()
testY=torch.tensor(test_data[:,:,0]).unsqueeze(2).float()

# best_model = torch.load('8-EC-speed/model/LSTM_SH_dist.pth')
# best_model = torch.load('8-EC-speed/model/LSTM_SH_dist_nospeed.pth')
# best_model = torch.load('8-EC-speed/model/LSTM_SZ_dist.pth')
# best_model = torch.load('8-EC-speed/model/LSTM_SZ_dist_nospeed.pth')
# best_model = torch.load('8-EC-speed/model/LSTM_CN_dist.pth')
# best_model = torch.load('8-EC-speed/model/LSTM_CN_dist_nospeed.pth')


with torch.no_grad():
    best_model.eval()  # 进入评估模式
    output = best_model(testX.to(device))
    predictions = output.squeeze().to('cpu').numpy()
    targets = testY.squeeze().to('cpu').numpy()
"""method1"""
# Metric1=np.array(evaluation(targets.reshape(-1), predictions.reshape(-1)))
# """method2=method1"""
# predictions_sum = np.sum(predictions, axis=1)
# targets_sum = np.sum(targets, axis=1)
# Metric2=np.array(evaluation(targets_sum, predictions_sum))

# Metric2=np.array(evaluation(np.sum(targets, axis=1, keepdims=True)
#                             , np.sum(predictions, axis=1, keepdims=True)))
# Metric2

#%%
"循环"

A=[]
for size in [256,300]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    input_size = 12  # 输入大小
    hidden_size = size  # 隐藏层大小
    num_layers = 2  # 层数
    batch_size = 16
    epochs = 300
    output_size = 1  # 输出大小
    lr = 0.0001

    LSTM_model = LSTM(input_size, hidden_size, num_layers,
                      output_size).to(device)
    optimizer = torch.optim.AdamW(LSTM_model.parameters(), lr=lr,
                                  betas=(0.9, 0.9999), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=5, factor=0.90)

    criterion = nn.MSELoss()

    best_model = train_model(LSTM_model, dataloaders_train, dataloaders_valid,
                             epochs, optimizer, criterion, batch_size)
    del LSTM_model
    testX = torch.tensor(test_data[:, :, 1:]).float()
    testY = torch.tensor(test_data[:, :, 0]).unsqueeze(2).float()
    test_dataset = Data.TensorDataset(testX, testY)
    dataloaders_test = Data.DataLoader(dataset=test_dataset,
                                       batch_size=16,
                                       shuffle=False,
                                       generator=torch.Generator().manual_seed(42))

    with torch.no_grad():
        best_model.eval()  # 进入评估模式
        all_simu = []
        all_real = []
        for i, (x, y) in enumerate(dataloaders_test):
            pred = best_model(x.to(device)).float()
            Norm_pred = pred.data.cpu().numpy()
            all_simu.append(Norm_pred.squeeze(2))
            all_real.append(y.squeeze(2))

        targets = np.vstack(all_real)
        predictions = np.vstack(all_simu)

    """method1"""
    Metric2 = np.array(evaluation(np.sum(targets, axis=1, keepdims=True)
                                  , np.sum(predictions, axis=1, keepdims=True)))
    A.append(Metric2)

B = np.array(A)
#%%
"""FOR dist plot"""
# 计算每个样本在第二维度上全是0的行的数量
testX=torch.tensor(test_data[:,:,1:]).float()
testY=torch.tensor(test_data[:,:,0]).unsqueeze(2).float()
zero_row_counts = (testX.sum(dim=2) == 0).sum(dim=1)
# 初始化六个类别
class_1_indices = []
class_2_indices = []
class_3_indices = []
class_4_indices = []
class_5_indices = []
class_6_indices = []

# 根据 zero_row_counts 将样本索引分配到六个类别中
for i, count in enumerate(zero_row_counts):
    if 0 <= count <= 200:
        class_1_indices.append(i)
    elif 200 < count <= 300:
        class_2_indices.append(i)
    elif 300 < count <= 350:
        class_3_indices.append(i)
    elif 350 < count <= 370:
        class_4_indices.append(i)
    elif 370 < count <= 380:
        class_5_indices.append(i)
    else:
        class_6_indices.append(i)

# 抽取对应的元素
class_1_predictions = [predictions[i] for i in class_1_indices]
class_2_predictions = [predictions[i] for i in class_2_indices]
class_3_predictions = [predictions[i] for i in class_3_indices]
class_4_predictions = [predictions[i] for i in class_4_indices]
class_5_predictions = [predictions[i] for i in class_5_indices]
class_6_predictions = [predictions[i] for i in class_6_indices]

class_1_targets = [targets[i] for i in class_1_indices]
class_2_targets = [targets[i] for i in class_2_indices]
class_3_targets = [targets[i] for i in class_3_indices]
class_4_targets = [targets[i] for i in class_4_indices]
class_5_targets = [targets[i] for i in class_5_indices]
class_6_targets = [targets[i] for i in class_6_indices]

class_1_sum_predictions = [np.sum(arr) for arr in class_1_predictions]
class_2_sum_predictions = [np.sum(arr) for arr in class_2_predictions]
class_3_sum_predictions = [np.sum(arr) for arr in class_3_predictions]
class_4_sum_predictions = [np.sum(arr) for arr in class_4_predictions]
class_5_sum_predictions = [np.sum(arr) for arr in class_5_predictions]
class_6_sum_predictions = [np.sum(arr) for arr in class_6_predictions]

class_1_sum_targets = [np.sum(arr) for arr in class_1_targets]
class_2_sum_targets = [np.sum(arr) for arr in class_2_targets]
class_3_sum_targets = [np.sum(arr) for arr in class_3_targets]
class_4_sum_targets = [np.sum(arr) for arr in class_4_targets]
class_5_sum_targets = [np.sum(arr) for arr in class_5_targets]
class_6_sum_targets = [np.sum(arr) for arr in class_6_targets]


Metric1=np.array(evaluation(np.array(class_1_sum_targets),
                            np.array(class_1_sum_predictions)))
Metric2=np.array(evaluation(np.array(class_2_sum_targets),
                            np.array(class_2_sum_predictions)))
Metric3=np.array(evaluation(np.array(class_3_sum_targets),
                            np.array(class_3_sum_predictions)))
Metric4=np.array(evaluation(np.array(class_4_sum_targets),
                            np.array(class_4_sum_predictions)))
Metric5=np.array(evaluation(np.array(class_5_sum_targets),
                            np.array(class_5_sum_predictions)))
Metric6=np.array(evaluation(np.array(class_6_sum_targets),
                            np.array(class_6_sum_predictions)))

Metrics = [Metric1[0], Metric2[0], Metric3[0], Metric4[0], Metric5[0], Metric6[0]]
Metrics_array = np.array(Metrics)
# datalen = np.array([len(class_1_indices), len(class_2_indices), len(class_3_indices),
#                     len(class_4_indices), len(class_5_indices), len(class_6_indices)])
Metrics_array
#%%

#%%
import shap


feature_list=["Speed","Travel time","Travel distance",
                          "Period","Date","DOD","SOH",
                          "Vol_std","Temp_std"]
# num_samples_to_select = 50
# indices = torch.randperm(trainX.size(0))[:num_samples_to_select]
# background_data = trainX[indices][:,:50,:].to(device)

explainer = shap.GradientExplainer(best_model, testX[:,30:31,:].to(device))
#
# shap_values = explainer(trainX)
# 创建一个 DeepExplainer 对象
# explainer = shap.Explainer(best_model, trainX)

best_model.train()  # 将模型切换到训练模式
shap_values = explainer(testX[:,30:31,:].to(device))
best_model.eval()
1
#%%
plt.rcParams['font.family'] = 'Times New Roman'

shap_values.feature_names=["Speed","Travel time","Travel distance",
                          "Period","Date","DOD","SOH",
                          "Vol_std","Temp_std"]
shap_values.data=shap_values.data.cpu()
shap_values1=shap_values[:,0,:]
# shap_values1=shap_values.values.reshape(-1,9)
# base_values=np.full(1000, -0.01)
# shap_values1.base_values=base_values

shap.plots.beeswarm(shap_values1,show=False)
plt.tight_layout()
# plt.savefig(r"8-EC-speed/FIGS/explainable-1.svg", dpi=600)
# plt.savefig(r"8-EC-speed/FIGS/explainable-31.svg", dpi=600)
# plt.savefig(r"8-EC-speed/FIGS/explainable-61.svg", dpi=600)
plt.show()
# shap.plots.heatmap(shap_values1)
# shap.plots.scatter(shap_values1[:, 1], color=shap_values)
# shap.plots.violin(shap_values[:,0,:], color="red")
#%%
shap.dependence_plot('Speed', shap_values1, testX[:,62:63,:].to(device),
                     # display_features=X_display,
                     interaction_index='Capital Gain')


#%%
cumulative_values = []

for array in CN_data:
    cumulative_sum = np.cumsum(array[:, 2])
    cumulative_values.append(cumulative_sum[-1])

cumulative_array = np.array(cumulative_values).reshape(-1, 1)
filtered_array = cumulative_array[(cumulative_array <= 10800) & (cumulative_array >= 900)]
np.mean(filtered_array)/60
#%%
cumulative_values = []

for array in CN_data:
    cumulative_sum = np.cumsum(array[:, 2])
    cumulative_values.append(cumulative_sum[-1])

cumulative_array = np.array(cumulative_values).reshape(-1, 1)
filtered_array = cumulative_array[(cumulative_array <= 10800) & (cumulative_array >= 900)]
np.mean(filtered_array)/60

#%%
import shap


feature_list=["Speed","Duration","Distance","Temperature","Wind speed"
                          "Period","Weekday","Season","DOD","SOH",
                          "Vol_std","Temp_std"]
# num_samples_to_select = 50
# indices = torch.randperm(trainX.size(0))[:num_samples_to_select]
# background_data = trainX[indices][:,:50,:].to(device)
explainer = shap.GradientExplainer(best_model, testX[:,0:1,:].to(device))
#
# shap_values = explainer(trainX)
# 创建一个 DeepExplainer 对象
# explainer = shap.Explainer(best_model, trainX)

best_model.train()  # 将模型切换到训练模式
shap_values = explainer(testX[:,0:1,:].to(device))
best_model.eval()
1
#%%
plt.rcParams['font.family'] = 'Times New Roman'

shap_values.feature_names=["Speed","Travel time","Travel distance",
                          "Period","Date","DOD","SOH",
                          "Vol_std","Temp_std"]
shap_values.data=shap_values.data.cpu()
shap_values1=shap_values[:,0,:]
# shap_values1=shap_values.values.reshape(-1,9)
# base_values=np.full(1000, -0.01)
# shap_values1.base_values=base_values

shap.plots.beeswarm(shap_values1,show=False)
plt.tight_layout()
# plt.savefig(r"8-EC-speed/FIGS/explainable-1.svg", dpi=600)
# plt.savefig(r"8-EC-speed/FIGS/explainable-31.svg", dpi=600)
# plt.savefig(r"8-EC-speed/FIGS/explainable-61.svg", dpi=600)
plt.show()