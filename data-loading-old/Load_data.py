import os
import random
import numpy as np
import tensorflow as tf
import pickle

train_split, val_split, test_split = 0.7, 0.2, 0.1
h, w = 220, 220
chn, batch_size = 3, 32
image_paths = []
# preset variables
# may need to change image size and batch size depending on the dataset
directory = ''
# add the path of where the compressed image files are located (unzip the file if not already done)

image_paths = []
for i in os.listdir(directory):
    if i.endswith('.jpg'):
        image_path = os.path.join(directory, i)
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, chn)
        img = tf.image.resize(img, size=(h, w))
        img = tf.image.convert_image_dtype(img, tf.float32)
        image_paths.append(img)
        
random.shuffle(image_paths)

num = len(image_paths)
num_train_images, num_val_images, num_test_images = int(num*train_split), int(num*val_split), int(num*test_split)

trdata = tf.data.Dataset.from_tensor_slices(image_paths[:num_train_images])
trdata = trdata.batch(batch_size)

vldata = tf.data.Dataset.from_tensor_slices(image_paths[num_train_images:num_train_images + num_val_images])
vldata = vldata.batch(batch_size)

testdata = tf.data.Dataset.from_tensor_slices(image_paths[-num_test_images:])
testdata = testdata.batch(batch_size)

# Convert the datasets to lists of NumPy arrays
trdata = list(trdata.as_numpy_iterator())
vldata = list(vldata.as_numpy_iterator())
testdata = list(testdata.as_numpy_iterator())

with open('train.pkl', 'wb') as i:
    pickle.dump(trdata, i)
with open('val.pkl', 'wb') as i:
    pickle.dump(vldata, i)
with open('test.pkl', 'wb') as i:
    pickle.dump(testdata, i)
# convert datasets to pickle files for easy access and less loading time
