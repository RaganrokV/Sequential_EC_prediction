# -*- coding: utf-8 -*-
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
import numpy as np
import pickle
from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 15,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config)
import pandas as pd
#%%

# Load from .pkl file
with open('8-EC-speed/route_data/ST_GRU_SH_ERR.pkl', 'rb') as f:
    ST_GRU_SH_ERR= pickle.load(f)

with open('8-EC-speed/route_data/ST_GRU_SZ_ERR.pkl', 'rb') as f:
    ST_GRU_SZ_ERR= pickle.load(f)

with open('8-EC-speed/route_data/ST_GRU_CN_ERR.pkl', 'rb') as f:
    ST_GRU_CN_ERR= pickle.load(f)

ST_GRU_SH_residual=ST_GRU_SH_ERR['REAL']-ST_GRU_SH_ERR['PRED']
ST_GRU_SZ_residual=ST_GRU_SZ_ERR['REAL']-ST_GRU_SZ_ERR['PRED']
ST_GRU_CN_residual=ST_GRU_CN_ERR['REAL']-ST_GRU_CN_ERR['PRED']

#%%
# 绘制散点图
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(8, 6))

plt.scatter(range(len(ST_GRU_SH_residual)), ST_GRU_SH_residual,
            color='#EA8379', edgecolor='grey', linewidths=0.5, alpha=0.9, label='SHEV dataset')

plt.scatter(range(len(ST_GRU_CN_residual)), ST_GRU_CN_residual,
            color='#7DAEE0', edgecolor='grey', linewidths=0.5,  alpha=0.9, label='CNEV dataset')
plt.scatter(range(len(ST_GRU_SZ_residual)), ST_GRU_SZ_residual,
            color='#B395BD', edgecolor='grey', linewidths=0.5,  alpha=0.9, label='SZEV dataset ')

plt.xlabel('Trip number', fontsize='20')
plt.ylabel('Residuals (kWh/trip)', fontsize='20')
plt.xticks(fontsize='15')
plt.yticks(fontsize='15')
plt.legend()
plt.grid(True)
plt.tight_layout()
# 保存图像
plt.savefig(r"8-EC-speed/FIGS/ERR_ST_GRU.svg", dpi=600)

# 显示图像
plt.show()
#%%
# Load from .pkl file
with open('8-EC-speed/route_data/SA_LSTM_SH_ERR.pkl', 'rb') as f:
    SA_LSTM_SH_ERR= pickle.load(f)

with open('8-EC-speed/route_data/SA_LSTM_SZ_ERR.pkl', 'rb') as f:
    SA_LSTM_SZ_ERR= pickle.load(f)

with open('8-EC-speed/route_data/SA_LSTM_CN_ERR.pkl', 'rb') as f:
    SA_LSTM_CN_ERR= pickle.load(f)

SA_LSTM_SH_residual=SA_LSTM_SH_ERR['REAL']-SA_LSTM_SH_ERR['PRED']
SA_LSTM_SZ_residual=SA_LSTM_SZ_ERR['REAL']-SA_LSTM_SZ_ERR['PRED']
SA_LSTM_CN_residual=SA_LSTM_CN_ERR['REAL']-SA_LSTM_CN_ERR['PRED']

#%%
# 绘制散点图
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(8, 6))



plt.scatter(range(len(SA_LSTM_SH_residual)), SA_LSTM_SH_residual,
            color='#EA8379', edgecolor='grey', linewidths=0.5, alpha=0.9, label='SHEV dataset')

plt.scatter(range(len(SA_LSTM_CN_residual)), SA_LSTM_CN_residual,
            color='#7DAEE0', edgecolor='grey', linewidths=0.5,  alpha=0.9, label='CNEV dataset')
plt.scatter(range(len(SA_LSTM_SZ_residual)), SA_LSTM_SZ_residual,
            color='#B395BD', edgecolor='grey', linewidths=0.5,  alpha=0.9, label='SZEV dataset ')
plt.xlabel('Trip number', fontsize='20')
plt.ylabel('Residuals (kWh/trip)', fontsize='20')
plt.xticks(fontsize='15')
plt.yticks(fontsize='15')
plt.legend()
plt.grid(True)
plt.tight_layout()
# 保存图像
plt.savefig(r"8-EC-speed/FIGS/ERR_SA_LSTM.svg", dpi=600)

# 显示图像
plt.show()