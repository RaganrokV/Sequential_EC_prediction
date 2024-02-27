# -*- coding: utf-8 -*-
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from My_utils.evaluation_scheme import evaluation
import pickle
import torch.utils.data as Data
import warnings
import torch.nn.functional as F
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
data = [arr for arr in SZ_data if len(arr) >= 10]
# data = [arr for arr in CN_data if len(arr) >= 10]
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
trainX=torch.tensor(train_data[:,:,1:]).transpose(1, 2).float()
trainY=torch.tensor(train_data[:,:,0]).unsqueeze(2).transpose(1, 2).float()
vaildX=torch.tensor(val_data[:,:,1:]).transpose(1, 2).float()
vaildY=torch.tensor(val_data[:,:,0]).unsqueeze(2).transpose(1, 2).float()


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
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


class TransformerTS(nn.Module):
    def __init__(self,
                 input_dim,
                 dec_seq_len,
                 out_seq_len,
                 d_model,
                 nhead,
                 num_encoder_layers,
                 num_decoder_layers,
                 dim_feedforward,
                 dropout,
                 activation,
                 custom_encoder=None,
                 custom_decoder=None):
        r"""
        Args:
            input_dim: dimision of imput series
            d_model: the number of expected features in the encoder/decoder inputs (default=512).
            nhead: the number of heads in the multiheadattention models (default=8).
            num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
            num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            dropout: the dropout value (default=0.1).
            activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
            custom_encoder: custom encoder (default=None).
            custom_decoder: custom decoder (default=None).


        """
        super(TransformerTS, self).__init__()
        self.transform = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder,
        )
        self.pos = PositionalEncoding(d_model)
        self.enc_input_fc = nn.Linear(input_dim, d_model)
        self.dec_input_fc = nn.Linear(input_dim, d_model)
        # self.out_fc = nn.Linear(dec_seq_len * d_model, out_seq_len)
        self.out_fc = nn.Linear(d_model, out_seq_len)
        self.dec_seq_len = dec_seq_len
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, out_seq_len)

    def forward(self, x):
        x = x.transpose(2 ,1)
        # embedding
        embed_encoder_input = self.pos(self.enc_input_fc(x))
        embed_decoder_input = self.dec_input_fc(x[-self.dec_seq_len:, :])
        # transform
        x = self.transform(embed_encoder_input, embed_decoder_input)

        # x = x.view(-1, self.dec_seq_len * x.size(-1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)


        return x

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

            loss = criterion(pre_y,y)
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
                val_loss += criterion(output,
                                      target).item()

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
# Initialize your Transformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

input_dim = 12  # 输入大小

dec_seq_len = 500
out_seq_len = 1
batch_size = 16
lr = 0.01

T_model = TransformerTS(input_dim,
                        dec_seq_len,
                        out_seq_len,
                        d_model=8,
                        nhead=4,
                        num_encoder_layers=2,
                        num_decoder_layers=2,
                        dim_feedforward=128,
                        dropout=0.1,
                        activation='relu',
                        custom_encoder=None,
                        custom_decoder=None).to(device)

optimizer = torch.optim.AdamW(T_model.parameters(), lr=lr,
                              betas=(0.9, 0.9999), weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       patience=5, factor=0.90)
criterion = nn.MSELoss()

best_model = train_model(T_model, dataloaders_train, dataloaders_valid,
                         10, optimizer, criterion, batch_size)


#%%
# torch.save(best_model, '8-EC-speed/model/TNN_SH_SA.pth')
# torch.save(best_model, '8-EC-speed/model/TNN_SH_SANV.pth')
# torch.save(best_model, '8-EC-speed/model/TNN_SZ_SA.pth')
# torch.save(best_model, '8-EC-speed/model/TNN_SZ_SANV.pth')
# torch.save(best_model, '8-EC-speed/model/TNN_CN_SA.pth')
# torch.save(best_model, '8-EC-speed/model/TNN_CN_SANV.pth')
#%%
"predict"
testX=torch.tensor(test_data[:,:,1:]).transpose(1, 2).float()
testY=torch.tensor(test_data[:,:,0]).unsqueeze(2).float()

test_dataset = Data.TensorDataset(testX, testY)
dataloaders_test= Data.DataLoader(dataset=test_dataset,
                                    batch_size=16,
                                   shuffle=False,
                                   generator=torch.Generator().manual_seed(42))


best_model.to(device).eval()
all_simu = []
all_real = []
for i, (x, y) in enumerate(dataloaders_test):
    with torch.no_grad():
        pred = best_model(x.to(device)).float()
        Norm_pred = pred.data.cpu().numpy()
        all_simu.append(Norm_pred.squeeze(2))
        all_real.append(y.squeeze(2))

targets=np.vstack(all_real)
predictions=np.vstack(all_simu)

"""method1"""
# Metric1=np.array(evaluation(targets.reshape(-1), predictions.reshape(-1)))
# """method2=method1"""
# predictions_sum = np.sum(predictions, axis=1)
# targets_sum = np.sum(targets, axis=1)
# Metric2=np.array(evaluation(targets_sum, predictions_sum))
Metric2=np.array(evaluation(np.sum(targets, axis=1, keepdims=True)
                            , np.sum(predictions, axis=1, keepdims=True)))
print("seg")

Metric2

#%%
A=[]
for LR in [0.1,0.01,0.001,0.0001,0.00001]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    input_dim = 12  # 输入大小

    dec_seq_len = 500
    out_seq_len = 1
    batch_size = 16
    lr = LR


    T_model = TransformerTS(input_dim,
                            dec_seq_len,
                            out_seq_len,
                            d_model=8,
                            nhead=4,
                            num_encoder_layers=2,
                            num_decoder_layers=2,
                            dim_feedforward=128,
                            dropout=0.1,
                            activation='relu',
                            custom_encoder=None,
                            custom_decoder=None).to(device)

    optimizer = torch.optim.AdamW(T_model.parameters(), lr=lr,
                                  betas=(0.9, 0.9999), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=5, factor=0.90)

    criterion = nn.MSELoss()

    best_model = train_model(T_model, dataloaders_train, dataloaders_valid,
                             100, optimizer, criterion, batch_size)
    del T_model
    testX = torch.tensor(test_data[:, :, 1:]).transpose(1, 2).float()
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




