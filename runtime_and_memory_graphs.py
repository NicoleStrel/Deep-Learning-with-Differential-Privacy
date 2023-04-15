import matplotlib.pyplot as plt
import numpy as np
import os

# base code written by Ritvik Jayanthi, optimized by Nicole Streltsov

def read_from_metric_files(folder, filenames_dict):
    '''
    read metrics from text file and add it to a dictionary in the form -->  key = x label, value = array of floats

    @param folder (string): folder to read from
    @param filenames_dict (dict): dictionary of text file key words and their corresponding x label
    @return metric_data (dict)
    '''
    metric_data = {}

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            file_path = os.path.join(folder, file)
            with open(file_path, 'r') as f:
                keys = [key for key in filenames_dict.keys() if key in file]
                if keys:
                    name = filenames_dict[keys[0]]
                    metric_data[name] = list(map(float, (line.split(': ')[1] for line in f)))
    return metric_data

def get_runtime_and_memory_lists(metric_data, x_labels):
    '''
    convert the metric dictionary into two lists for runtime and memory bars

    @param metric_data (dictionary): metrics in the form --> key = x label, value = array of floats
    @param x_labels (list): string labels for the x axis
    @return runtime, memory (lists)
    '''
    runtime = []
    memory = []
    for label in x_labels:
        runtime.append(metric_data[label][0])
        memory.append(metric_data[label][1])
    return runtime, memory

def gen_plot(axs, axis_idx, x_labels, legend_labels, metric_data, bar_width, data):
    '''
    creates double bar graph plot where one bar is for memory, the other for runtime.

    @param axs: matplotlib axs object
    @param axis_idx (int): index of the subplot to create
    @param x_labels (list): string labels for the x axis
    @param legend_labels (list): legend labels
    @param metric_data (dict): metrics in the form --> key = x label, value = array of floats
    @param bar_width (float): the width of each bar
    @param data (string): string for the title of the plot
    @return none
    '''
    x_pt = np.arange(len(x_labels))
    runtime, memory = get_runtime_and_memory_lists(metric_data, x_labels)

    axs[axis_idx].bar(x_pt, runtime, width=bar_width, label=legend_labels[0], color='#d4afb9')
    axs[axis_idx].bar(x_pt + bar_width, memory, width=bar_width, label=legend_labels[1], color='#7ec4cf')
    axs[axis_idx].set_xticks(x_pt + bar_width / 2)
    axs[axis_idx].set_xticklabels(x_labels)
    axs[axis_idx].set_xlabel('Framework')
    axs[axis_idx].set_title('Runtime and Memory for the ' + data + ' X-ray Dataset')
    axs[axis_idx].legend()

def create_combined_plots(chest_folder, knee_folder, filenames_dict, x_labels):
    '''
    creates 2 subplots to plot the memory/runtime double bar graphs for the chest and knee X-ray datasets

    @param chest_folder (string): path for the chest data folder
    @param knee_folder (string): path for the knee data folder
    @param filenames_dict (dict): dictionary of text file key words and their corresponding x label
    @param x_labels (list): string labels for the x axis
    @return none
    '''

    #collect the data from the text files
    metric_data_chest = read_from_metric_files(chest_folder, filenames_dict)
    metric_data_knee = read_from_metric_files(knee_folder, filenames_dict)

    print(metric_data_chest)

    # define plot values
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30, 5))
    legend_labels = ['Runtime (s)', 'Memory (MB)']
    bar_width = 0.2

    # create plots
    gen_plot(axs, 0, x_labels, legend_labels, metric_data_chest, bar_width, 'Chest')
    gen_plot(axs, 1, x_labels, legend_labels, metric_data_knee, bar_width, 'Knee')
    plt.show()

if __name__ == '__main__':
    chest_folder = 'metrics/chest/'
    knee_folder = 'metrics/knee/'
    x_labels = ['SGD\nTF Objax', 'DP-SGD\nTF Objax', 'SGD\nTF Keras', 'DP-SGD-JL\nTF Keras', 'SGD\nPyTorch', 'DP-SGD-FL\nPyTorch', 'PATE\nPyTorch']
    filenames_dict = {'federated_dp': 'DP-SGD-FL\nPyTorch', 'non_federated': 'SGD\nPyTorch', 'tf_objax_DP_SGD': 'DP-SGD\nTF Objax', 'tf_objax_regular': 'SGD\nTF Objax', 'dp_pate': 'PATE\nPyTorch', 'tf_keras_DP_SGD': 'DP-SGD-JL\nTF Keras', 'tf_keras_SGD': 'SGD\nTF Keras'}

    create_combined_plots(chest_folder, knee_folder, filenames_dict, x_labels)
