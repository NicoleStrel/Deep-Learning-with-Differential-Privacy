@@ -0,0 +1,36 @@
# dp-knee-xray-images-dataset


Data downloaded from https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity?select=auto_test

**Notes:**
 - 80% train, 10% validation, 10% test data split
 - 9786 images total, 3857 - normal, 1770 - doubtful, 2578 - minimal, 1286 - moderate, 295 - severe
 - image crops to (64, 64)
 - rgb colour mode
 - final compressed pickle file stores numpy arrays
 - does data augmentation on the train data, so that the labels are balanced. There are 7829 images in the training data.

**How to use:**
- download the files and put them into a directory eg. 'knee-data'
- add the following code to obtain the values:

```
import pickle
import gzip
path = 'knee-data' #os.path.join(os.path.dirname(os.getcwd()), 'knee-data')
with gzip.open(os.path.join(path, 'knee_x_train.gz'), 'rb') as i:
    x_train = pickle.load(i)
with gzip.open(os.path.join(path, 'knee_x_val.gz'), 'rb') as i:
    x_valid = pickle.load(i)    
with gzip.open(os.path.join(path, 'knee_x_test.gz'), 'rb') as i:
    x_test = pickle.load(i)  
with gzip.open(os.path.join(path, 'knee_y_train.gz'), 'rb') as i:
    y_train = pickle.load(i)  
with gzip.open(os.path.join(path, 'knee_y_val.gz'), 'rb') as i:
    y_valid = pickle.load(i) 
with gzip.open(os.path.join(path, 'knee_y_test.gz'), 'rb') as i:
    y_test = pickle.load(i)
```
