import matplotlib.pyplot as plt
import numpy as np

# written by Ritvik Jayanthi

file_paths = [
    'federated_dp_chest.txt',
    'non_federated_chest.txt',
    'tf_objax_DP_SGD_chest.txt',
    'tf_objax_regular_SGD_chest.txt'
]

federated_dp_chest, non_federated_chest, tf_objax_DP_SGD_chest, tf_objax_regular_SGD_chest = None, None, None, None

for i, file_path in enumerate(file_paths):
    arr = np.loadtxt(file_path)
    if i == 0:
        federated_dp_chest = arr
    elif i == 1:
        non_federated_chest = arr
    elif i == 2:
        tf_objax_DP_SGD_chest = arr
    elif i == 3:
        tf_objax_regular_SGD_chest = arr
        
file_paths = [
    'federated_dp_knee.txt',
    'non_federated_knee.txt',
    'tf_objax_DP_SGD_knee.txt',
    'tf_objax_regular_SGD_knee.txt'
]

federated_dp_knee, non_federated_knee, tf_objax_DP_SGD_knee, tf_objax_regular_SGD_knee = None, None, None, None

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



fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
legend_labels = ['Runtime (s)', 'Memory (MB)']

x_labels_knee = ['SGD_F\nfed', 'DP_SGD_F\nfed', 'DP_SGD\nDP', 'SGD\nDP']
bar_width = 0.3
x_pt_knee = np.arange(len(x_labels_knee))
runtime_knee = [federated_dp_knee[0], non_federated_knee[0], tf_objax_DP_SGD_knee[0], tf_objax_regular_SGD_knee[0]]
memory_knee = [federated_dp_knee[1], non_federated_knee[1], tf_objax_DP_SGD_knee[1], tf_objax_regular_SGD_knee[1]]
axs[0].bar(x_pt_knee, runtime_knee, width=bar_width, label=legend_labels[0], color='#d4afb9')
axs[0].bar(x_pt_knee + bar_width, memory_knee, width=bar_width, label=legend_labels[1], color='#7ec4cf')
axs[0].set_xticks(x_pt_knee + bar_width / 2)
axs[0].set_xticklabels(x_labels_knee)
axs[0].set_xlabel('Models')
axs[0].set_title('Knee Model Comparison')
axs[0].legend()

x_labels_chest = ['SGD_F\nfed', 'DP_SGD_F\nfed', 'DP_SGD\nDP', 'SGD\nDP']
bar_width = 0.3
x_pt_chest = np.arange(len(x_labels_chest))
runtime_chest = [federated_dp_chest[0], non_federated_chest[0], tf_objax_DP_SGD_chest[0], tf_objax_regular_SGD_chest[0]]
memory_chest = [federated_dp_chest[1], non_federated_chest[1], tf_objax_DP_SGD_chest[1], tf_objax_regular_SGD_chest[1]]
axs[1].bar(x_pt_chest, runtime_chest, width=bar_width, label=legend_labels[0], color='#d4afb9')
axs[1].bar(x_pt_chest + bar_width, memory_chest, width=bar_width, label=legend_labels[1], color='#7ec4cf')
axs[1].set_xticks(x_pt_chest + bar_width / 2)
axs[1].set_xticklabels(x_labels_chest)
axs[1].set_xlabel('Models')
axs[1].set_title('Chest Model Comparison')
axs[1].legend()
plt.show()
