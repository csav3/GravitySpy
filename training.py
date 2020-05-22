import theano
from keras import backend as K
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, Conv2D
from itertools import combinations
from itertools import permutations
import random

import numpy as np
import os
import pandas
os.chdir('/Users/carlynsavino/Desktop/training2')


trainingset = pandas.read_pickle('trainingset_100_images_each_class.pkl') 


classes = []
# Adds class names from each sample to a list (includes duplicates)
for i in range(len(trainingset.index)):
    classes.append(trainingset.iloc[i,5])

# Removes duplicate class names
classes = np.unique(classes) 

# create a variable for number of classes used
num_classes = 3

# All possible combinations of 3 classes
list_combos = list(combinations(classes, num_classes)) # order matters; no repeated elements

# randomly select 1 for now
# create a variable for number of combos
num_combos = 1

combos_random = random.sample(list_combos, num_combos)

for i in range(1, 4):
    tset_labels = []
    tset_data = []
    for combos in combos_random:
        for index, c in enumerate(combos):
            rows_match = trainingset.loc[trainingset['true_label'] == c]
            half_sec_img = np.vstack(rows_match.iloc[:,0].values)
            one_sec_img = np.vstack(rows_match.iloc[:,1].values)
            two_sec_img = np.vstack(rows_match.iloc[:,2].values)
            three_sec_img = np.vstack(rows_match.iloc[:,3].values)

            half_sec_img_reshape = half_sec_img.reshape(-1, 140, 170,1)
            one_sec_img_reshape = one_sec_img.reshape(-1, 140, 170,1)
            two_sec_img_reshape = two_sec_img.reshape(-1, 140, 170,1)
            three_sec_img_reshape = three_sec_img.reshape(-1, 140, 170,1)

            tset_labels.extend([index] * len(rows_match)) #extend keeps a single list and extends it
            tset_data.append(np.concatenate((half_sec_img_reshape, one_sec_img_reshape, two_sec_img_reshape, three_sec_img_reshape), axis = i))
        tset_data_array = np.vstack(tset_data)
        print(tset_data_array.shape)
        tset_labels_array = np.asarray(tset_labels)
        # print(tset_labels_array.shape)
        
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
    # if 3 layers is true, call function that does only 3 layers.
    # if not, do 4 layers

        model.compile(loss = 'sparse_categorical_crossentropy',
                    optimizer='adadelta',
                    metrics=['accuracy'])
     

    #fit using the training data and training labels

        fit = model.fit(tset_data_array, tset_labels_array, validation_split = 0.15, epochs = 1)
    
# Print table information at the end of each loop 
    # save the results to something - look up how to do this; save training results to file
    # A table with: Number of layers: num_layers
                    # Classes: num_classes
                    # Number of samples: num_samples
                    # Training score: train_acc
                    # Validation score: val_acc

