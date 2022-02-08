#!/usr/bin/anaconda3/bin/python3
"""
CODE from:: https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py

Trains a ResNet on the CIFAR10 dataset.
ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, TensorBoard, LambdaCallback
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model,load_model
from keras.datasets import cifar10
import numpy as np
import os
import sys
import random
sys.path.append("../..") 
sys.path.append(os.getcwd())
import tensorflow.keras.backend as K
from time import time
from utils import *
from config import conf
from model import *
from triplet_model import triplet_net, triplet_loss
from sklearn.preprocessing import normalize
# Loading interations:



# Training parameters
data_augmentation = conf.use_data_augmentation
num_classes = conf.num_classes
input_shape = [conf.img_dim_1, conf.img_dim_2, conf.img_dim_3]


# Subtracting pixel mean improves accuracy
subtract_pixel_mean = False

if conf.img_dim_1 >= 224:
    n = 27
elif conf.img_dim_1 >= 128:
    n = 18
else:
    n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1


# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2


'''Model creat or load'''



for _ in range(conf.interation):

    # get_training_data_by_anatation_or_simulation
    os.system('cd ../../Active_Selection_Truncate && python sample_selection.py')

    annotator="general"



    if load_interation()==0:
        if version == 2:
            net = resnet_v2(input_shape=input_shape, depth=depth, num_classes=num_classes)
        else:
            net = resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)
        
        model = triplet_net(net, num_classes)
    else:
        model = load_model('saved_models/'+str(annotator)+'_%s_%s.h5' % ((load_interation()-1),conf.dataset), custom_objects={'triplet_loss': triplet_loss})                 
        print ('\n','load ',str(annotator),'model','%s_%s.h5' % ((load_interation()-1),conf.dataset),'\n')

    model.summary()

    # began_to_train
    print ('\n','In interation',str(load_interation()),'training '+str(annotator)+' model','\n')


    # input data

    training_data, training_batch_num = get_data(True,data_augmentation,annotator)
    testing_data, testing_batch_num = get_data(False,data_augmentation,annotator)

    
    # compile training
    # model.compile(loss=triplet_loss, optimizer=Adam(lr=conf.learning_rate,clipnorm=1., clipvalue=0.5))
    model.compile(loss=triplet_loss, optimizer=Adam(lr=lr_schedule(0),clipnorm=1., clipvalue=0.5))
    



    '''Prepare model model saving directory and callbacks'''

    #save model

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = str(annotator)+'_%s_%s.h5' % ((load_interation()),conf.dataset)
    # model_name = '%s_%s_%s_model_epoch-{epoch:03d}_loss-{loss:0.4f}.h5' % (load_interation(),conf.dataset, model_type)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                monitor='loss',
                                verbose=1,
                                save_best_only=True,
                                period=10)


    # reduce the learning rate

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(monitor='loss',
                               factor=0.1,
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)



    # visualize by tensorboard
    # tensorboard = TensorBoard(log_dir="logs/{}".format(model_type))
    tensorboard = TensorBoard(log_dir="logs/{}".format(model_name))

    callbacks = [checkpoint, lr_reducer,lr_scheduler,tensorboard]
    # callbacks = [checkpoint]




    '''Run training, with or without data augmentation.'''
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit_generator(training_data, epochs=conf.epoch, steps_per_epoch=training_batch_num, validation_data=testing_data,validation_steps=testing_batch_num,verbose=2,shuffle=True,callbacks=callbacks)

    else:
        print('Using real-time data augmentation.')
        model.fit_generator(generator=training_data, steps_per_epoch=1560, epochs=conf.epoch, 
                            shuffle=False, use_multiprocessing=True,callbacks=callbacks)





    # # if option 2, we need to difine the layer.
    net=model.get_layer(index=4)

    '''Get the embeddings'''

    train=True

    training_image, _, _ ,_ = load_data(train,annotator)
    embeddings_train = []
    for i in range(0, len(training_image), conf.batch_size):
        start = i
        end = start + conf.batch_size

        feed = normalize_image(training_image[start:end])
        # print(feed.shape)
        
        pred = net.predict(feed, batch_size=conf.batch_size, verbose=2)
        # print(pred)
        for p in pred:
            embeddings_train.append(p)
        
    # print(random.sample(embeddings_train,5))

    embeddings_train = np.array(embeddings_train)
    save_embeddings(train,embeddings_train,annotator)


    
    train=False

    testing_image, _, _ ,_ = load_data(train,annotator)
    embeddings_test = []
    for i in range(0, len(testing_image), conf.batch_size):
        start = i
        end = start + conf.batch_size

        feed = normalize_image(testing_image[start:end])
        # print(feed.shape)
        
        pred = net.predict(feed, batch_size=conf.batch_size, verbose=2)
        # print(pred)
        for p in pred:
            embeddings_test.append(p)
        
    # print(random.sample(embeddings_test,5))

    embeddings_test = np.array(embeddings_test)
    save_embeddings(train,embeddings_test,annotator)




    annotators=["Annotator_1","Annotator_2","Annotator_3"]

    for annotator in annotators:

        if load_interation()==0:
            if version == 2:
                net = resnet_v2(input_shape=input_shape, depth=depth, num_classes=num_classes)
            else:
                net = resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)
            
            model = triplet_net(net, num_classes)
        else:
            model = load_model('saved_models/'+str(annotator)+'_%s_%s.h5' % ((load_interation()-1),conf.dataset), custom_objects={'triplet_loss': triplet_loss})                 
            print ('\n','load ',str(annotator),'model','%s_%s.h5' % ((load_interation()-1),conf.dataset),'\n')


        

        # began_to_train
        print ('\n','In interation',str(load_interation()),'training '+str(annotator)+' model','\n')


        # input data

        training_data, training_batch_num = get_data(True,data_augmentation,annotator)
        testing_data, testing_batch_num = get_data(False,data_augmentation,annotator)

        
        # compile training
        # model.compile(loss=triplet_loss, optimizer=Adam(lr=conf.learning_rate,clipnorm=1., clipvalue=0.5))
        model.compile(loss=triplet_loss, optimizer=Adam(lr=lr_schedule(0),clipnorm=1., clipvalue=0.5))
        



        '''Prepare model model saving directory and callbacks'''

        #save model

        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = str(annotator)+'_%s_%s.h5' % ((load_interation()),conf.dataset)
        # model_name = '%s_%s_%s_model_epoch-{epoch:03d}_loss-{loss:0.4f}.h5' % (load_interation(),conf.dataset, model_type)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)
        # Prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(filepath=filepath,
                                    monitor='loss',
                                    verbose=1,
                                    save_best_only=True,
                                    period=10)


        # reduce the learning rate

        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(monitor='loss',
                                factor=0.1,
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-6)



        # visualize by tensorboard
        # tensorboard = TensorBoard(log_dir="logs/{}".format(model_type))
        tensorboard = TensorBoard(log_dir="logs/{}".format(model_name))

        callbacks = [checkpoint, lr_reducer,lr_scheduler,tensorboard]
        # callbacks = [checkpoint]




        '''Run training, with or without data augmentation.'''
        if not data_augmentation:
            print('Not using data augmentation.')
            model.fit_generator(training_data, epochs=conf.epoch, steps_per_epoch=training_batch_num, validation_data=testing_data,validation_steps=testing_batch_num,verbose=2,shuffle=True,callbacks=callbacks)

        else:
            print('Using real-time data augmentation.')
            model.fit_generator(generator=training_data, steps_per_epoch=1560, epochs=conf.epoch, 
                                shuffle=False, use_multiprocessing=True,callbacks=callbacks)





        # if option 2, we need to difine the layer.
        net=model.get_layer(index=4)

        '''Get the embeddings'''

        train=True

        training_image, _, _ ,_ = load_data(train,annotator)
        embeddings_train = []
        for i in range(0, len(training_image), conf.batch_size):
            start = i
            end = start + conf.batch_size

            feed = normalize_image(training_image[start:end])
            # print(feed.shape)
            
            pred = net.predict(feed, batch_size=conf.batch_size, verbose=2)
            # print(pred)
            for p in pred:
                embeddings_train.append(p)
            
        # print(random.sample(embeddings_train,5))

        embeddings_train = np.array(embeddings_train)
        save_embeddings(train,embeddings_train,annotator)


        
        train=False

        testing_image, _, _ ,_ = load_data(train,annotator)
        embeddings_test = []
        for i in range(0, len(testing_image), conf.batch_size):
            start = i
            end = start + conf.batch_size

            feed = normalize_image(testing_image[start:end])
            # print(feed.shape)
            
            pred = net.predict(feed, batch_size=conf.batch_size, verbose=2)
            # print(pred)
            for p in pred:
                embeddings_test.append(p)
            
        # print(random.sample(embeddings_test,5))

        embeddings_test = np.array(embeddings_test)
        save_embeddings(train,embeddings_test,annotator)







    # Next interation

    interation=load_interation()
    interation +=1

    save_interation(interation)




