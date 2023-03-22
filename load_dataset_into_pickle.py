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

def count_num_classes(ds):
    '''
    Counts the number of classes in a tenorflow dataset
    @return int counts
    '''
    # get labels
    labels_np = np.array(list(ds.map(lambda x, y: y).as_numpy_iterator()))

    # use np.unique to find counts
    counts = {label: count for label, count in zip(*np.unique(labels_np, return_counts=True))}

    print("num classes --->", counts)
    return counts

def augment_data(image, label):
    '''
    Lambda function that data augments the provided images. Does the following physical tranformations randomly:
        - flips horizontally
        - flips vertically 
        - rotates 0, 90, 180 or 360 degrees (one of)
    '''
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.image.random_flip_up_down(image)
    return image, label

def extend_dataset(ds):
    '''
    Ensures that the dataset is balanced - ie. each label has enough data points to compare with each other. 

    @param tensorflow ds
    @return tensorflow ds
    '''

    print("length of original -->", len(list(ds)))
    original_size = len(list(ds))
    counts = count_num_classes(ds)
    max_samples = max(counts.values())

    for label, count in counts.items(): #loop through (to work on different # of classes)
        dataset_extend = ds.filter(lambda x, y: tf.equal(y, label))
        dataset_extend = dataset_extend.repeat((max_samples - count) // count + 1)
        dataset_extend = dataset_extend.map(augment_data, num_parallel_calls=tf.data.experimental.AUTOTUNE) #parallize image augmentation
        dataset_extend = dataset_extend.take(max_samples - count)
        ds = ds.concatenate(dataset_extend)
    ds = ds.shuffle(buffer_size=original_size, seed=1)

    print("length of augmented dataset -->", len(list(ds)))
    count_num_classes(ds)

    return ds

def get_dataset_np_arrays(directory, classifications, size, val_split, data_augment = False):
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

    if data_augment:
        train_ds = extend_dataset(train_ds)

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
    size = (64, 64)
    val_split = 0.2
    classifications = [] #order of numerical labels (starting from 0)
    directory = '' # path where image files are located

    if which_dataset == "chest":
        # dataset stats: 5856 images, 1584 - normal, 4274 - pneumonia
        print("loading the chest dataset ...")

        classifications = ["NORMAL", "PNEUMONIA"]
        directory = 'chest_xray'
        folder_to_save = "chest-data-new"
        data_augment = True

        x_train, x_val, x_test, y_train, y_val, y_test = get_dataset_np_arrays(directory, classifications, size, val_split, data_augment)
        dump_to_pickle_files(folder_to_save, which_dataset, x_train, x_val, x_test, y_train, y_val, y_test)

    elif which_dataset == "knee":
        # dataset stats: 9786 images, 3857 - normal, 1770 - doubtful, 2578 - minimal, 1286 - moderate, 295 - severe
        print("loading the knee dataset ...")
        
        classifications = ["NORMAL", "DOUBTFUL", "MINIMAL", "MODERATE", "SEVERE"]
        directory = 'knee_xray'
        folder_to_save = "knee-data-new"
        data_augment = True
        
        x_train, x_val, x_test, y_train, y_val, y_test = get_dataset_np_arrays(directory, classifications, size, val_split, data_augment)
        dump_to_pickle_files(folder_to_save, which_dataset, x_train, x_val, x_test, y_train, y_val, y_test)
    else:
        print ("choose a datset from: 'chest' or 'knee'")
