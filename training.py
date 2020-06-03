import theano
from keras import backend as K
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, Conv2D
from itertools import combinations
from itertools import permutations
import random
import math

import numpy as np
import os
import pandas

# function for rounding down; used to get number of samples
def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

# reads in dataset; path may differ on your own machine
trainingset = pandas.read_pickle('GravitySpyData/pickeleddata/trainingset_100_images_each_class.pkl') 

classes = []

# adds class names from each sample to a list (includes duplicates)
for i in range(len(trainingset.index)):
    classes.append(trainingset.iloc[i,5])

# removes duplicate class names
classes = np.unique(classes) 

# create a variable for number of classes used
num_classes = 3

# all possible combinations of 3 classes
list_combos = list(combinations(classes, num_classes)) # order matters; no repeated elements

# create a variable for number of combos
num_combos = 2

combos_random = random.sample(list_combos, num_combos)

# clears the csv file for new program run-through
if os.path.exists('metrics.csv'):
    os.remove('metrics.csv')

# creates pandas dataframe with column names
results = pandas.DataFrame(columns=['num_combos', 'num_layers', 'concat_axis', 'classes', 'num_samples', 'num_val_samples', 'accuracy', 'val_accuracy'])

for i in range(1, 4):
    tset_labels = []
    tset_data = []
    for combos in combos_random:
        classes_used = []
        for index, c in enumerate(combos):
            rows_match = trainingset.loc[trainingset['true_label'] == c]
            half_sec_img = np.vstack(rows_match.iloc[:,0].values)
            one_sec_img = np.vstack(rows_match.iloc[:,1].values)
            two_sec_img = np.vstack(rows_match.iloc[:,2].values)
            three_sec_img = np.vstack(rows_match.iloc[:,3].values)

            classes_used.append(c)

            half_sec_img_reshape = half_sec_img.reshape(-1, 140, 170,1)
            one_sec_img_reshape = one_sec_img.reshape(-1, 140, 170,1)
            two_sec_img_reshape = two_sec_img.reshape(-1, 140, 170,1)
            three_sec_img_reshape = three_sec_img.reshape(-1, 140, 170,1)

            tset_labels.extend([index] * len(rows_match)) #extend keeps a single list and extends it
            tset_data.append(np.concatenate((half_sec_img_reshape, one_sec_img_reshape, two_sec_img_reshape, three_sec_img_reshape), axis = i))

        tset_data_array = np.vstack(tset_data)
        print(tset_data_array.shape)
        tset_labels_array = np.asarray(tset_labels)

        classes_used = np.array(classes_used)

        model = Sequential()

        model.add(Conv2D(16, (3,3), input_shape= tset_data_array.shape[1:4])) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(32, (3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, (3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, (3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dense(4, activation = 'softmax'))

        
        model.summary()

        model.compile(loss = 'sparse_categorical_crossentropy',
                    optimizer='adadelta',
                    metrics=['accuracy'])
     

        # portion of data used for validation
        val_amount = 0.15

        # fit using the training data and training labels
        fit = model.fit(tset_data_array, tset_labels_array, validation_split = val_amount, epochs = 1)

        # obtain number of samples
        num_samples = round_down((1 - val_amount) * len(tset_data_array))

        # obtain number of layers
        num_layers = len(model.layers) 

        # fill in pandas dataframe
        results = results.append({'num_combos': num_combos,
                                    'num_layers': num_layers,
                                    'concat_axis': i,
                                    'classes': classes_used,
                                    'num_samples': num_samples,
                                    'num_val_samples': len(tset_data_array) - num_samples,
                                    'accuracy': fit.history['accuracy'],
                                    'val_accuracy': fit.history['val_accuracy']
                                    }, ignore_index=True)

# save results to csv file
results.to_csv('metrics.csv', mode = 'a', header = True)
