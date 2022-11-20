# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:42:40 2022

@author: Administrator
"""

import tensorflow as tf
from keras.models import *




model.compile(
    
                loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                metrics = ['binary_accuracy']
                
             )








checkpoint = tf.keras.callbacks.ModelCheckpoint(
    
                                                    filepath=".../",
                                                    monitor='val_loss',
                                                    metrics=['acc'],
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode='min'
                                                    
                                                )







history = model.fit_generator(   
    
                            generator = train_gen, 
                            steps_per_epoch = 1000,
                            validation_data = validate_gen, 
                            validation_steps = 400, 
                            epochs =50,
                            callbacks=[checkpoint]
                            
                         )