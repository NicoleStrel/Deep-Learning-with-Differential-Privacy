# dp-chest-xray-images-dataset


Data downloaded from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

**Notes:**
 - 80% train, 10% validation, 10% test data split
 - 5856 images total, 1584 - normal, 4274 - pneumonia
 - image crops to (64, 64)
 - rgb colour mode
 - final compressed pickle file stores numpy arrays
 - does data augmentation on the train data, so that the labels are balanced. There are 6808 images in the training data.

**How to use:**
- download the files and put them into a directory eg. 'chest-data'
- add the following code to obtain the values:

```
import pickle
import gzip

path = 'chest-data' #os.path.join(os.path.dirname(os.getcwd()), 'chest-data')

with gzip.open(os.path.join(path, 'chest_x_train.gz'), 'rb') as i:
    x_train = pickle.load(i)
with gzip.open(os.path.join(path, 'chest_x_val.gz'), 'rb') as i:
    x_valid = pickle.load(i)    
with gzip.open(os.path.join(path, 'chest_x_test.gz'), 'rb') as i:
    x_test = pickle.load(i)  
with gzip.open(os.path.join(path, 'chest_y_train.gz'), 'rb') as i:
    y_train = pickle.load(i)  
with gzip.open(os.path.join(path, 'chest_y_val.gz'), 'rb') as i:
    y_valid = pickle.load(i) 
with gzip.open(os.path.join(path, 'chest_y_test.gz'), 'rb') as i:
    y_test = pickle.load(i)
```
