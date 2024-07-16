# -*- coding: utf-8 -*-
import random
import numpy as np
from My_utils.evaluation_scheme import evaluation
import pickle
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
#%%

#%%
with open('6-imbalanced_EC/itinerary_data/SH_trip.pkl', 'rb') as f:
    SH_data = pickle.load(f)

with open('6-imbalanced_EC/itinerary_data/SZ_trip.pkl', 'rb') as f:
    SZ_data = pickle.load(f)

with open('6-imbalanced_EC/itinerary_data/CN_trip.pkl', 'rb') as f:
    CN_data = pickle.load(f)

CN_data[CN_data == '#DIV/0!'] = 0
CN_data = CN_data.astype(float)
#%%

#%%
# correlation_matrix = np.corrcoef(data1, rowvar=False)
# print(correlation_matrix)

#%%
# Assuming data1, data2, and data3 are already defined as numpy arrays
data1 = np.vstack(SH_data)[:,1:]
data2 = np.vstack(SZ_data)[:,1:]
data3 = np.vstack(CN_data)[:,1:]

# Define variable names
variable_names = ['avg_speed', 'expected_time', 'expected_distance', 'T', 'wind',
                  'period', 'day_of_week', 'season', 'DOD', 'SOH', 'vol_std', 'temp_std']

# Function to calculate VIF for each dataset
def calculate_vif(data, variable_names):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = variable_names
    vif_data["VIF"] = [variance_inflation_factor(data, i) for i in range(data.shape[1])]
    return vif_data

# Calculate VIF for each dataset
vif_data1 = calculate_vif(data1, variable_names)
vif_data2 = calculate_vif(data2, variable_names)
vif_data3 = calculate_vif(data3, variable_names)

# Combine results into a single DataFrame
df_vif = pd.concat([vif_data1, vif_data2['VIF'], vif_data3['VIF']], axis=1)
df_vif.columns = ['Variable', 'VIF_data1', 'VIF_data2', 'VIF_data3']

# Display the DataFrame with VIF values
print(df_vif)
