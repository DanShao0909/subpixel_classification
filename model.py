# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:29:46 2022

@author: Administrator
"""

import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.applications.vgg16 import VGG16


class vgg_unet():
    
    def __init__(self):
        
        self.h=512
        self.w=512
        
        
        
    def vgg_512(self):
        
        base_model = VGG16(weights='imagenet',include_top=False,input_shape=(512,512,3))
        model1 = Model(inputs=base_model.get_layer('block1_conv1').input, outputs=base_model.get_layer('block1_pool').output)
        
        return model1
    
    
    
    
    def vgg_256(self):
        
        base_model = VGG16(weights='imagenet',include_top=False,input_shape=(512,512,3))
        model2 = Model(inputs=base_model.get_layer('block2_conv1').input, outputs=base_model.get_layer('block2_pool').output)
        
        return model2
    
    
    
    
    def vgg_128(self):
        
        base_model = VGG16(weights='imagenet',include_top=False,input_shape=(512,512,3))
        model3 = Model(inputs=base_model.get_layer('block3_conv1').input, outputs=base_model.get_layer('block3_pool').output)
        
        return model3
    
    
    
    def vgg_64(self):
        
        base_model = VGG16(weights='imagenet',include_top=False,input_shape=(512,512,3))
        model4 = Model(inputs=base_model.get_layer('block4_conv1').input, outputs=base_model.get_layer('block4_pool').output)
        
        return model4
    
    
    
    
    def unet(self):
        
        inputShape = (512, 512, 1)
        inputs = tf.keras.layers.Input(shape=inputShape)
        
        # D1
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(inputs)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        y1 = x 
        x = tf.keras.layers.Conv2D(64, (3, 3), 2, padding='same')(x) #(None, 256, 256, 64)

        y1_1 = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(y1) #(None, 512, 512, 3)
        y1_1 = self.vgg_512()(y1_1) 

        x = tf.concat([x,y1_1],-1) 
        y11 = x




        # D2
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        y2 = x
        x = tf.keras.layers.Conv2D(128, (3, 3), 2, padding='same')(x) 

        y2_2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(y2)
        y2_2 = self.vgg_256()(y2_2)
      
        x = tf.concat([x,y2_2],-1) 
        y22 = x
        
        
        
        
        # D3
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        y3 = x 
        x = tf.keras.layers.Conv2D(256, (3, 3), 2, padding='same')(x)

        y3_3 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(y3) 
        y3_3 = self.vgg_128()(y3_3)

        x = tf.concat([x,y3_3],-1) 
        y33 = x
        
        
        
        
        # D4
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        y4 = x 
        x = tf.keras.layers.Conv2D(512, (3, 3), 2, padding='same')(x) 

        y4_4 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(y4) 
        y4_4 = self.vgg_64()(y4_4) 

        x = tf.concat([x,y4_4],-1) 
        y44 = x
        
        
        
        
        # 5
        x = tf.keras.layers.Conv2D(1024, (3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(1024, (3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = Dropout(0.5)(x)
    
    
    
        
        # U4
        x = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(x)
        print(x.shape)
        x = tf.concat([x,y4],-1)
        print(x.shape)
        
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        
        
        
        # U3
        x = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
        print(x.shape)
        x = tf.concat([x,y3],-1)
        print(x.shape)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        
        
        
        
        # U2
        x = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        print(x.shape)
        x = tf.concat([x,y2],-1)
        print(x.shape)
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        
        
        
        # U1
        x = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        print(x.shape)
        x = tf.concat([x,y1],-1)
        print(x.shape)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(1, (1, 1), padding='same')(x)
        x = tf.keras.layers.Activation('sigmoid')(x)
        
        
        
        
        model = tf.keras.models.Model(inputs, x)
        
        
        
        return model
