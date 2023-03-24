import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
urls = [
    'https://github.com/NicoleStrel/Deep-Learning-with-Differential-Privacy/blob/main/metrics/chest/federated_dp_chest.txt?raw=true',
    'https://github.com/NicoleStrel/Deep-Learning-with-Differential-Privacy/blob/main/metrics/chest/non_federated_chest.txt?raw=true',
    'https://github.com/NicoleStrel/Deep-Learning-with-Differential-Privacy/blob/main/metrics/chest/tf_objax_DP_SGD_chest.txt?raw=true',
    'https://github.com/NicoleStrel/Deep-Learning-with-Differential-Privacy/blob/main/metrics/chest/tf_objax_regular_SGD_chest.txt?raw=true'
]

federated_dp_chest, non_federated_chest, tf_objax_DP_SGD_chest, tf_objax_regular_SGD_chest = None, None, None, None

for i, url in enumerate(urls):
    arr = np.loadtxt(url)
    if i == 0:
        federated_dp_chest = arr
    elif i == 1:
        non_federated_chest = arr
    elif i == 2:
        tf_objax_DP_SGD_chest = arr
    elif i == 3:
        tf_objax_regular_SGD_chest = arr

"""

federated_dp_chest = [538.6092126369476, 829.671875]
non_federated_chest = [381.4143555164337, 1100.0859375]
tf_objax_DP_SGD_chest = [267.57230043411255, 6028.98828125]
tf_objax_regular_SGD_chest = [123.68997192382812, 5909.16015625]

x_labels = ['SGD_F\nfed', 'DP_SGD_F\nfed', 'DP_SGD\nDP', 'SGD\nDP']
bar_width = 0.3
x_pt = np.arange(len(x_labels))
runtime = [federated_dp_chest[0], non_federated_chest[0], tf_objax_DP_SGD_chest[0], tf_objax_regular_SGD_chest[0]]
memory = [federated_dp_chest[1], non_federated_chest[1], tf_objax_DP_SGD_chest[1], tf_objax_regular_SGD_chest[1]]
fig, ax = plt.subplots()

ax.bar(x_pt, runtime, width=bar_width, label='Runtime', color='#d4afb9')
ax.bar(x_pt + bar_width, memory, width=bar_width, label='Memory', color='#7ec4cf')

ax.set_xticks(x_pt + bar_width / 2)
ax.set_xticklabels(x_labels)
ax.set_xlabel('Models')
ax.set_title('Chest Model Comparison')
ax.legend()
plt.show()
