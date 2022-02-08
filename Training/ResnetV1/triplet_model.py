from model import resnet_v2, resnet_v1
from config import conf
from keras.layers import Lambda, Input, Concatenate
from keras.models import Model
import tensorflow as tf
import keras.backend as K

def triplet_net(net, num_classes=10):
    

    generated = Input(shape=(3,conf.img_dim_1, conf.img_dim_2, conf.img_dim_3), name='input')

    anchor  = Lambda(lambda x: x[:,0])(generated)
    pos     = Lambda(lambda x: x[:,1])(generated)
    neg     = Lambda(lambda x: x[:,2])(generated)

    # print('anchor', anchor)
    # print('postitive', pos)
    # print('negative', neg)

    # Get the embedded values
    anchor_embedding    = net(anchor)
    pos_embedding       = net(pos)
    neg_embedding       = net(neg)

    # print('anchor_embedding', anchor_embedding)
    # print('pos_embedding', pos_embedding)
    # print('neg_embedding', neg_embedding)

    merged_output = Concatenate()([anchor_embedding, pos_embedding, neg_embedding])

    model = Model(inputs=generated, outputs=merged_output)

    return model

def triplet_loss(y_true, y_pred):

    # anchor, positive, negative = tf.split(y, num_or_size_splits=3, axis=1)        
    # print('anchor.shape',   anchor.shape)
    # print('positive.shape', positive.shape)
    # print('negative.shape', negative.shape)

    # pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    # neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    # basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), conf.margin)
    # loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    # print('loss', loss)

    
   
    anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)        
    # print('anchor.shape',   anchor.shape)
    # print('positive.shape', positive.shape)
    # print('negative.shape', negative.shape)




    pos_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1))
    neg_dist1 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1))
    neg_dist2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(positive, negative)), 1))



    new_margin=tf.add(conf.margin,tf.divide(y_true, conf.increase_margin_rate))


    basic_loss1 = tf.add(tf.subtract(pos_dist, neg_dist1), new_margin)
    basic_loss2 = tf.add(tf.subtract(pos_dist, neg_dist2), new_margin)
    loss1 = tf.reduce_mean(tf.maximum(basic_loss1, 0.0), 0)
    loss2 = tf.reduce_mean(tf.maximum(basic_loss2, 0.0), 0)
    loss=(tf.add(loss1,loss2))/2

    # print('loss', loss)




    return loss