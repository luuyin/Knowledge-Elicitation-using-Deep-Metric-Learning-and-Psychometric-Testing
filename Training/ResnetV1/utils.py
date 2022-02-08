import tensorflow as tf
import numpy as np
import os, json, sys
sys.path.append('../..')
from config import conf
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator



train_path = os.path.join('..', '..', 'data', conf.dataset, 'train')

test_path = os.path.join('..', '..', 'data', conf.dataset, 'test')

def load_interation():
    """
    load_interations
    """
    with open(os.path.join('..', '..', 'interation.txt'), "r") as f:    
        interation = f.read()   
        interation=int(interation)
    return interation


def save_interation(data):
    """
    save_interations
    """
    with open(os.path.join('..', '..', 'interation.txt'),"w") as f:
            f.write(str(data))  

        
def load_data(train,annotator):
    """
    :return:    images, names and triplets (only names)
    """

    if train:
        path = train_path
    else:
        path=test_path

    with open(os.path.join(path, conf.dataset + '_img_names.json')) as f:
        names = json.load(f)
    
    with open(os.path.join(path, conf.dataset + '_decoded_' + str(conf.img_dim_1) + 'x' + str(conf.img_dim_2) + '.json')) as t:
        images = json.load(t)
        images = np.array(images)

    with open(os.path.join(path, str(annotator)+'_'+conf.dataset + '_triplets_'+str(load_interation())+'.json')) as t:
        trip = json.load(t)
    triplets_index = np.array(trip)

    with open(os.path.join(train_path, str(annotator)+'_'+conf.dataset + '_distances_'+str(load_interation())+'.json')) as t:
        distances = json.load(t)   
  

    
    print('triplets_index.shape', triplets_index.shape)
    # print('images.shape', images.shape)

    return images, names, triplets_index,distances




def save_embeddings(train,data,annotator):
    if train:
        path = train_path
    else:
        path=test_path

    with open(os.path.join(path, str(annotator)+'_'+conf.dataset + '_embeddings_' + str(conf.img_dim_1) + 'x' + str(conf.img_dim_2) +'_'+ str(load_interation()) + '.json'), 'w') as outfile:
            print('embedding size ', len(data), len(data.tolist()))
            json.dump(data.tolist(), outfile)






"""
https://github.com/noelcodella/tripletloss-keras-tensorflow/blob/master/tripletloss.py
Use code above to implement data augmentation for tripletlosss
"""


def fitDataGen(data):
    datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
    
    datagen.fit(data)
    
    return datagen

# generator function for data augmentation
def createDataGen(images, names, triplets_index):

    datagen = fitDataGen(images / 255.)
    y = 1 * np.zeros(len(triplets_index))

    triplets = []
    for triplet_index in triplets_index:
        triplet = [[],[],[]]

        triplet[0] = images[names.index(triplet_index['Anchor'])]
        triplet[1] = images[names.index(triplet_index['positive'])]
        triplet[2] = images[names.index(triplet_index['negative'])]
        triplet = np.array(triplet).astype(int)

        triplets.append(triplet)
    
    # normalize data 
    triplets = np.array(triplets) / 255.

    # This will do preprocessing and realtime data augmentation:
    local_seed = 1337
    genAnchor = datagen.flow(triplets[:,0], y, batch_size=conf.batch_size, seed=local_seed, shuffle=False)
    genPos = datagen.flow(triplets[:,1], y, batch_size=conf.batch_size, seed=local_seed, shuffle=False)
    genNeg = datagen.flow(triplets[:,2], y, batch_size=conf.batch_size, seed=local_seed, shuffle=False)
    while True:
        A = genAnchor.next()
        P = genPos.next()
        N = genNeg.next()

        # change order from [3, batch, dim, dim, dim] 
        # to [batch, 3, dim, dim, dim]
        triplet = []
        for i in range(len(A[0])):
            _triplet = []    
            _triplet.append(A[0][i])
            _triplet.append(P[0][i])
            _triplet.append(N[0][i])
            triplet.append(_triplet)
        
        triplet = np.array(triplet)

        yield triplet, N[1]
            
def data_generator(names, images, triplets_index, batch_size, epoch,distances):
    """
    https://stackoverflow.com/questions/28507052/how-to-split-numpy-array-in-batches
    :param batch_size:  size of each group
    :param epoch:       amount of epochs
    :return:            Yields successive batch-sized lists from data.
    """
    
    while True:
       
     
        
        for i in range(0, len(triplets_index), batch_size):
            batch_triplets_index = triplets_index[i:i+batch_size]
            
            batch_distances      = distances[i:i+batch_size]  
            if len(batch_triplets_index) < batch_size:
                continue

            triplets = []
            for triplet_index in batch_triplets_index:
                triplet = [[],[],[]]

                triplet[0] = images[names.index(triplet_index['Anchor'])]
                triplet[1] = images[names.index(triplet_index['positive'])]
                triplet[2] = images[names.index(triplet_index['negative'])]
                triplet = np.array(triplet).astype(int)

                triplets.append(triplet)
            
            # normalize data 
            triplets = np.array(triplets) / 255.
            yield triplets, batch_distances
            
'''            # make sure non of the images contain none values           
            for batch in triplets:
                for trip in batch:
                    for image in trip:
                        for pixel in image:
                            for color in pixel:
                                if color is None:
                                    continue

            # keras expects a label (we dont use this in the loss function so can be anything)'''
           

def get_data(train,augmentation,annotator):

    images, names, triplets_index, distances = load_data(train,annotator)

    num_tr_batch = len(triplets_index) // conf.batch_size
    
    if augmentation:
        batch_triplets = createDataGen(images, names, triplets_index)
    else:
        batch_triplets = data_generator(names, images, triplets_index, conf.batch_size, conf.epoch,distances)

    return batch_triplets, num_tr_batch





def normalize_image(images):
    return np.array(images) / 255.

