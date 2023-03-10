import os
from PIL import Image
from multiprocessing import Pool
import matplotlib.pyplot as plt

def get_image_sizes(filename):
    with Image.open(filename) as img:
        return img.size

def read_image_sizes(filenames):
    with Pool(processes=8) as pool: #parallize the computatation with 8 threads
        sizes = pool.map(get_image_sizes, filenames)
    widths, heights = zip(*sizes)
    return widths, heights

def create_dataset_scatter(dir, img_extension, colors, plt_title):
    # go through the folders
    i = 0
    for folder in os.listdir(dir):
        path = os.path.join(dir, folder)
        filenames = [os.path.join(path, filename) for filename in os.listdir(path) if filename.endswith(img_extension)]
        widths, heights = read_image_sizes(filenames)
        plt.scatter(widths, heights, color=colors[i], label=folder, s=2)
        i+=1

    # finalize the scatter plot
    plt.title(plt_title)
    plt.xlabel('X-ray Image Width (px)')
    plt.ylabel('X-ray Image Height (px)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    dataset = "chest"
    colors = ["#eb9678", "#78bdeb"]
    img_extension = 'jpeg'

    if (dataset == "chest"):
        print("visualizing chest datset ...")
        dir = 'chest_xray'
        plt_title = "Chest X-ray Image Size Distribution"
        create_dataset_scatter(dir, img_extension, colors, plt_title)
    elif (dataset == "knee"):
        print("visualizing knee datset ...")
    else:
        print("choose an available dataset from: 'chest' or 'knee'")
