# -*- coding: utf-8 -*-
import pickle
import warnings

from scipy.interpolate import lagrange
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns

from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 15,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config)
#%%
"""读取"""
for i in range(1, 21):
    globals()['SH_V{}'.format(i)] = \
        pd.read_pickle('5-energy_estimation/EV_data/V{}.pkl'.
                       format(i), compression=None)

#%%
Weather = pd.read_csv('6-Route_EC_prediction/weather.csv')
#%%
def SH_trip(car_data, soc_min, soc_max):
    """filtering data"""
    SH_trips = car_data[(car_data['vehicledata_vehiclestatus'] == 1) &
                        (car_data['vehicledata_runmodel'] == 1) &
                        (car_data['vehicledata_chargestatus'] == 3) &
                        (car_data['vehicledata_soc'] >= soc_min) &
                        (car_data['vehicledata_soc'] <= soc_max)].assign(trip_number=0)

    SH_trips['time_diff'] = SH_trips['collectiontime'].diff() > pd.Timedelta(minutes=15)
    SH_trips['trip_number'] = SH_trips['time_diff'].cumsum()
    SH_trips = SH_trips.drop(columns=['time_diff'])
    """trips at least last 30min or soc will not change"""

    valid_trips = [trip_data for _, trip_data in SH_trips.groupby('trip_number') if len(trip_data) >= 180]

    return valid_trips
#%%
Itinerary=[]
for i in range(1, 21):
    Itinerary+=SH_trip(globals()['SH_V{}'.format(i)], 0,100)

print("finish")
#%%
# import random
# import matplotlib.pyplot as plt
#
# # 设置随机种子
# random.seed(42)
#
# # 从Itinerary中随机选择9个行程
# selected_itineraries = random.sample(Itinerary, 9)
#
# # 创建子图
# fig, axes = plt.subplots(3, 3, figsize=(12, 12))
#
# # 遍历选定的行程并绘制速度曲线
# for i, itinerary in enumerate(selected_itineraries):
#     # 提取速度数据
#     speeds = itinerary['vehicledata_speed'].values
#
#     # 绘制速度曲线
#     row = i // 3
#     col = i % 3
#     axes[row, col].plot(speeds)
#     axes[row, col].set_title(f"Itinerary {i + 1}")
#     axes[row, col].set_xlabel("Time")
#     axes[row, col].set_ylabel("Speed")
#     axes[row, col].set_ylim(0, 100)
#
# # 调整子图之间的间距
# plt.tight_layout()
#
# # 显示图形
# plt.show()

#%%
for i in range(1, 21):
    del globals()['SH_V{}'.format(i)]
#%%
"""formatting dataset"""
data=[]
for i in range(len(Itinerary)):
    """EC / KWH"""
    EC=(Itinerary[i]['vehicledata_soc'].iloc[0]-Itinerary[i]['vehicledata_soc'].iloc[-1])*50.6/100

    """speed"""
    avespeed = []
    grouped = Itinerary[i].groupby('vehicledata_summileage')

    for mileage, group in grouped:
        avg_speed = group[group['vehicledata_speed'] != 0]['vehicledata_speed'].mean()  # 排除值为0的数据点
        avespeed.append(avg_speed)

    route_speed=np.array(avespeed)
    """expected distance"""
    expected_distance=(Itinerary[i]['vehicledata_summileage'].iloc[-1]-
                       Itinerary[i]['vehicledata_summileage'].iloc[0])

    """expected time"""
    expected_time=int(len(Itinerary[i])/6)

    """period"""
    start_time = pd.to_datetime(Itinerary[i]['collectiontime'].iloc[0])

    hour = start_time.hour

    if 7 <= hour < 9:
        period = 1
    elif 17 <= hour < 19:
        period = 2
    else:
        period = 3

    """Temperature"""

    date = start_time.date()

    date_str = date.strftime('%Y-%m-%d')
    # 根据日期在天气数据中查找对应的行
    weather_row = Weather[Weather['日期'] == date_str]

    # 提取最高温度和最低温度
    highest_temp = weather_row['最高温度'].iloc[0]
    lowest_temp = weather_row['最低温度'].iloc[0]

    #
    raw_data = np.concatenate(([EC, expected_distance, expected_time, period, highest_temp, lowest_temp], route_speed))
    data.append(raw_data)
#%%
with open('6-Route_EC_prediction/data.pkl', 'wb') as f:
    pickle.dump(data, f)
#%%
# 获取数据长度
data_lengths = [len(row) for row in data]

# 绘制直方图
plt.hist(data_lengths, bins=100)  # 设置直方图的柱子数量
plt.xlabel('Data Length')
plt.ylabel('Frequency')
plt.title('Distribution of Data Lengths')
plt.show()
#%%
plt.boxplot(data_lengths)
plt.xlabel('Data')
plt.ylabel('Length')
plt.title('Boxplot of Data Lengths')
plt.show()
#%%
"""formatting dataset"""
data_segment=[]
for i in range(len(Itinerary)):
    """EC / KWH"""
    EC=(Itinerary[i]['vehicledata_soc'].iloc[0]-Itinerary[i]['vehicledata_soc'].iloc[-1])*50.6/100

    """speed"""
    df = Itinerary[i]
    df1 = df[df['vehicledata_speed'] != 0].copy()

    df1['time_diff'] = df1['collectiontime'].diff().dt.total_seconds()

    df1.reset_index(drop=True, inplace=True)
    breakpoints = df1.index[df1['time_diff'] >= 60].tolist()
    breakpoints.append(len(df1))

    segments = np.split(df1, breakpoints)

    segment_speeds = [segment['vehicledata_speed'].mean() for segment in segments]

    route_speed = np.array(segment_speeds[:-1])

    """expected distance"""
    expected_distance=(Itinerary[i]['vehicledata_summileage'].iloc[-1]-
                       Itinerary[i]['vehicledata_summileage'].iloc[0])

    """expected time"""
    expected_time=int(len(Itinerary[i])/6)

    """period"""
    start_time = pd.to_datetime(Itinerary[i]['collectiontime'].iloc[0])

    hour = start_time.hour

    if 7 <= hour < 9:
        period = 1
    elif 17 <= hour < 19:
        period = 2
    else:
        period = 3

    """Temperature"""

    date = start_time.date()

    date_str = date.strftime('%Y-%m-%d')
    # 根据日期在天气数据中查找对应的行
    weather_row = Weather[Weather['日期'] == date_str]

    # 提取最高温度和最低温度
    highest_temp = weather_row['最高温度'].iloc[0]
    lowest_temp = weather_row['最低温度'].iloc[0]

    #
    raw_data = np.concatenate(([EC, expected_distance, expected_time, period, highest_temp, lowest_temp], route_speed))
    data_segment.append(raw_data)


#%%
with open('6-Route_EC_prediction/data_segment.pkl', 'wb') as f:
    pickle.dump(data_segment, f)

#%%
# 获取数据长度
data_lengths = [len(row) for row in data_segment]

# 绘制直方图
plt.hist(data_lengths, bins=100)  # 设置直方图的柱子数量
plt.xlabel('Data Length')
plt.ylabel('Frequency')
plt.title('Distribution of Data Lengths')
plt.show()
#%%
plt.boxplot(data_lengths)
plt.xlabel('Data')
plt.ylabel('Length')
plt.title('Boxplot of Data Lengths')
plt.show()

#%%
"""formatting dataset  link path"""
link_data=[]
for i in range(len(Itinerary)):

    grouped = Itinerary[i].groupby('vehicledata_summileage')

    avespeed = []  # 存储每个分组的平均速度
    times = []  # 存储每个分组的时间
    energies = []  # 存储每个分组的能耗

    for mileage, group in grouped:
        # 计算平均速度
        avg_speed = group[group['vehicledata_speed'] != 0]['vehicledata_speed'].mean()  # 排除值为0的数据点
        avespeed.append(avg_speed)

        # 计算时间
        time = len(group) * 10 / 60  # 数据采样间隔为10秒
        times.append(time)

        # 计算能耗
        voltage = group['vehicledata_sumvoltage']
        current = group['vehicledata_sumcurrent']
        energy = (voltage * current * 10).sum() / 1000 / 3600
        energies.append(energy)

    route_speed = np.array(avespeed)
    route_time = np.array(times)
    route_energy = np.array(energies)

    """period"""
    start_time = pd.to_datetime(Itinerary[i]['collectiontime'].iloc[0])

    hour = start_time.hour

    if 7 <= hour < 9:
        period = 1
    elif 17 <= hour < 19:
        period = 2
    else:
        period = 3

    route_period = np.full_like(route_energy, period)

    """Temperature"""

    date = start_time.date()

    date_str = date.strftime('%Y-%m-%d')
    # 根据日期在天气数据中查找对应的行
    weather_row = Weather[Weather['日期'] == date_str]

    # 提取最高温度和最低温度
    highest_temp = weather_row['最高温度'].iloc[0]
    lowest_temp = weather_row['最低温度'].iloc[0]

    route_highest_temp = np.full_like(route_energy, highest_temp)
    route_lowest_temp = np.full_like(route_energy, lowest_temp)

    """distance"""

    route_distance = np.full_like(route_energy, 1000)

    # 假设这些数组的形状分别为 (m,)，可以通过 np.expand_dims 进行维度扩展
    route_speed = np.expand_dims(route_speed, axis=1)
    route_time = np.expand_dims(route_time, axis=1)
    route_energy = np.expand_dims(route_energy, axis=1)
    route_period = np.expand_dims(route_period, axis=1)
    route_highest_temp = np.expand_dims(route_highest_temp, axis=1)
    route_lowest_temp = np.expand_dims(route_lowest_temp, axis=1)
    route_distance = np.expand_dims(route_distance, axis=1)

    # 将数组按列拼接
    combined_array = np.column_stack((route_energy,route_speed, route_time,  route_period,
                                      route_highest_temp, route_lowest_temp, route_distance))

    link_data.append(combined_array)
 #%%
with open('6-Route_EC_prediction/link_data.pkl', 'wb') as f:
    pickle.dump(link_data, f)

