import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
import gzip

# written by Nicole Streltsov March 2023

def convert_to_np_arrays(tf_ds):
    '''
    Converts tensorflow dataset into numpy arrays for x (image ndarray) and y (classification)

    @param tf_ds: tensorflow dataset
    @return final_x numpy array, and finaly_y numpy array
    '''
    arr_x = []
    arr_y = []
    for x,y in tfds.as_numpy(tf_ds):
        arr_x.append(x)
        arr_y.append(np.array([y]))
    
    final_x = np.stack(arr_x)
    final_y = np.stack(arr_y)

    return final_x, final_y

def get_dataset_np_arrays(directory, classifications, size, val_split):
    '''
    Gets the numpy arrays of the dataset for train, test, and validation sets. 
    It splits the data via 80%:10%:10% split (train:val:test)

    Note: image_dataset_from_directory() function assumes that the directory is the 'main directory'
    and it loops through the contents of subdirectories which are the classes. The folder structure should look like:
        /directory/
            |__ class_1/
            |__ class_2/
            ...

    @param directory (string): directory name that holds the images folders
    @param classifcations (list): list of claffication labels that will be numerically labelled starting from 0
    @param size (tuple): image size to resize to
    @param val_split (double): fraction to split the data into sets
    @returns numpy arrays for x_train, x_val, x_test, y_train, y_val, y_test
    '''

    # get the training data - 80% from the directory files
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        class_names=classifications,
        color_mode='rgb',
        batch_size=None,
        image_size=size,
        shuffle=True,
        seed=1,
        validation_split=val_split,
        subset="training",
    )

    # get the validation data - temporarly 20%, as we will split it into test in the next step
    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        class_names=classifications,
        color_mode='rgb',
        batch_size=None,
        image_size=size,
        shuffle=True,
        seed=1,
        validation_split=val_split,
        subset="validation",
    )

    # get the test data - divide the validation data in half, so its 10% each 
    val_ds_size = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_ds_size // 2)
    val_ds = val_ds.skip(val_ds_size // 2)

    # now, get the numpy arrays
    x_train, y_train = convert_to_np_arrays(train_ds)
    x_val, y_val = convert_to_np_arrays(val_ds)
    x_test, y_test = convert_to_np_arrays(test_ds)

    print("numpy array shapes: ", x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape)
    
    return x_train, x_val, x_test, y_train, y_val, y_test
    
def dump_to_pickle_files(folder, prefix, x_train, x_val, x_test, y_train, y_val, y_test):
    """
    convert numpy arrays to pickle files for easy access and less loading time

    @param x_train, x_val, x_test, y_train, y_val, y_test numpy arrays
    @return None
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    with gzip.open(os.path.join(folder, prefix + '_x_train.gz'), 'wb') as i:
        pickle.dump(x_train, i)
    with gzip.open(os.path.join(folder, prefix +'_x_val.gz'), 'wb') as i:
        pickle.dump(x_val, i)
    with gzip.open(os.path.join(folder, prefix +'_x_test.gz'), 'wb') as i:
        pickle.dump(x_test, i)

    with gzip.open(os.path.join(folder, prefix +'_y_train.gz'), 'wb') as i:
        pickle.dump(y_train, i)
    with gzip.open(os.path.join(folder, prefix +'_y_val.gz'), 'wb') as i:
        pickle.dump(y_val, i)
    with gzip.open(os.path.join(folder, prefix +'_y_test.gz'), 'wb') as i:
        pickle.dump(y_test, i)

if __name__ == '__main__':

    # parameters
    which_dataset = "chest"
    size = (256, 256)
    val_split = 0.2
    classifications = [] #order of numerical labels (starting from 0)
    directory = '' # path where image files are located

    if which_dataset == "chest":
        # dataset stats: 5856 images, 1584 - normal, 4274 - pneumonia
        print("loading the chest dataset ...")

        classifications = ["NORMAL", "PNEUMONIA"]
        directory = 'chest_xray'
        folder_to_save = "chest-data"

        x_train, x_val, x_test, y_train, y_val, y_test = get_dataset_np_arrays(directory, classifications, size, val_split)
        dump_to_pickle_files(folder_to_save, which_dataset, x_train, x_val, x_test, y_train, y_val, y_test)

    elif which_dataset == "knee":
        print("loading the knee dataset ...")

    else:
        print ("choose a datset from: 'chest' or 'knee'")
