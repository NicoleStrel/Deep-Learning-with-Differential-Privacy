import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
urls = [
    'https://github.com/NicoleStrel/Deep-Learning-with-Differential-Privacy/blob/main/metrics/knee/federated_dp_knee.txt?raw=true',
    'https://github.com/NicoleStrel/Deep-Learning-with-Differential-Privacy/blob/main/metrics/knee/non_federated_knee.txt?raw=true',
    'https://github.com/NicoleStrel/Deep-Learning-with-Differential-Privacy/blob/main/metrics/knee/tf_objax_DP_SGD_knee.txtraw=true',
    'https://github.com/NicoleStrel/Deep-Learning-with-Differential-Privacy/blob/main/metrics/knee/tf_objax_regular_SGD_knee.txt?raw=true'
]

federated_dp_chest, non_federated_chest, tf_objax_DP_SGD_chest, tf_objax_regular_SGD_chest = None, None, None, None

for i, url in enumerate(urls):
    arr = np.loadtxt(url)
    if i == 0:
        federated_dp_knee = arr
    elif i == 1:
        non_federated_knee = arr
    elif i == 2:
        tf_objax_DP_SGD_knee = arr
    elif i == 3:
        tf_objax_regular_SGD_knee = arr

"""

federated_dp_knee = [2169.327261686325, 1314.78125]
non_federated_knee = [789.095287322998, 2126.38671875]
tf_objax_DP_SGD_knee = [380.26026463508606, 6129.375]
tf_objax_regular_SGD_knee = [279.8356511592865, 5959.734375]

x_labels = ['SGD_F\nfed', 'DP_SGD_F\nfed', 'DP_SGD\nDP', 'SGD\nDP']
bar_width = 0.3
x_pt = np.arange(len(x_labels))
runtime = [federated_dp_knee[0], non_federated_knee[0], tf_objax_DP_SGD_knee[0], tf_objax_regular_SGD_knee[0]]
memory = [federated_dp_knee[1], non_federated_knee[1], tf_objax_DP_SGD_knee[1], tf_objax_regular_SGD_knee[1]]
fig, ax = plt.subplots()

ax.bar(x_pt, runtime, width=bar_width, label='Runtime', color='#d4afb9')
ax.bar(x_pt + bar_width, memory, width=bar_width, label='Memory', color='#7ec4cf')

ax.set_xticks(x_pt + bar_width / 2)
ax.set_xticklabels(x_labels)
ax.set_xlabel('Models')
ax.set_title('Knee Model Comparison')
ax.legend()
plt.show()
