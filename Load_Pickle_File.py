import pickle

#how to load data from pickle file

with open('name_to_load.pkl', 'rb') as i:
    name_the_file = pickle.load(i)
