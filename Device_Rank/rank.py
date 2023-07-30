import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append('../utilities/')
from graphGenerationUtilities import *
data_path = '/home/mabon/Cross_EM/cnn_output/stm/ID/same_device/TrainS2_TESTONSAMELOC/model10/rank_dir/ranking_raw_data.npz'
x1, y1 = a1['x'], a1['y']
data_path = '/home/mabon/Cross_EM/cnn_output/stm/ID/same_device/Train_S2_L11_Test_MULLOC/11/rank_dir/ranking_raw_data.npz'
a1 = np.load(data_path)
x2, y2 = a1['x'], a1['y']
data_path = '/home/mabon/Cross_EM/cnn_output/stm/ID/same_device/TrainS2_TESTONSAMELOC/model12/rank_dir/ranking_raw_data.npz'
a1 = np.load(data_path)
x3, y3 = a1['x'], a1['y']
fig, ax = plt.subplots()
#ax.plot(x4, y4,linewidth=2,label='S1_K1_200k')

ax.plot(x1[0:2000], y1[0:2000],linewidth=2,label='S2: Train_10_Test_10' ,marker='^',markevery=500,markersize=15)
#ax.plot(x2[0:2000], y2[0:2000],linewidth=2,label='S2: Train_11_Test_11',marker='o',markevery=500,markersize=10)
ax.plot(x3[0:2000], y3[0:2000],linewidth=2,label='S2: Train_12_Test_12',marker='s',markevery=500,markersize=10)
#ax.plot(x4[0:2000], y4[0:2000],linewidth=2,label='Automatic (h=8)',marker='*',markevery=500,markersize=10)


#ax.plot(x5[0:5000], y5[0:5000],linewidth=2,label='S2_K3_O2 (Offset -220)',marker='p',markevery=1000,markersize=10)


legend_without_duplicate_labels(ax, loc=1)
plt.rcParams['font.size'] = '16'
x_ticks = list(range(0, 2001, 500))
plt.xticks(x_ticks)
y_ticks = list(range(0, 258, 64))
plt.yticks(y_ticks)
plt.xlabel('No. of Test Traces')
plt.ylabel('Mean Rank')
plt.show()
