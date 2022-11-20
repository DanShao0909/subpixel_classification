# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:32:18 2022

@author: Administrator
"""
import cv2
import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from PIL import Image




Imgpath = []  
LabPath = [] 




for i in range(len(img_ids)):

    im00  = '.../' + img_ids[i]
    lb00 =  '.../' + img_ids[i]

    Imgpath.append(im00) 
    LabPath.append(lb00)
    
    
    
    
    

aug = tf.keras.preprocessing.image.ImageDataGenerator(
    
                                                        rotation_range = 360, 
                                                        zoom_range = 0.2, 
                                                        width_shift_range = 0.2, 
                                                        height_shift_range = 0.2, 
                                                        shear_range = 0.2, 
                                                        horizontal_flip = True,
                                                        fill_mode = "reflect"
                                                        
                                                        )






def train_image_generator(image_path,LabPath,st,ed,batch_size,aug = None): 
    
    nowinx = st 
    
    h = 512
    w = 512
    
    
    while True:
        
        im_array = []
        lb_array = []
 
        
        for i in range(batch_size):
 
            im = cv2.imread(image_path[nowinx],0)
            im = cv2.resize(im,(h,w),3)
            
            tmp_im_array = np.array(im)/255 
            tmp_im_array = np.asanyarray(tmp_im_array,dtype = np.float16) 
            tmp_im_array = tmp_im_array.reshape((h,w,1))
            tmp_im_array = tmp_im_array[np.newaxis,:,:,:] 
            
            
            lb = cv2.imread(LabPath[nowinx],0)
            lb = cv2.resize(lb,(h,w),3)
            lb = np.reshape(lb,(h,w,1))

            
            tmp_lb_array = np.asanyarray(lb,dtype = np.float16)
            tmp_lb_array[tmp_lb_array>0]=1
            tmp_lb_array= tmp_lb_array[np.newaxis,:,:,:]

            
            
            if len(im_array) == 0:
                
                im_array = tmp_im_array
                lb_array = tmp_lb_array 
                
            else:
                
                im_array = np.concatenate((im_array,tmp_im_array),axis=0) 
                lb_array = np.concatenate((lb_array,tmp_lb_array),axis=0) 
                
            nowinx = st if nowinx==ed else nowinx+1 
             
        

        if aug is not None : 
            
            new_array = im_array
            new_array = np.concatenate([new_array,lb_array],axis=-1) 
            new_array = next(aug.flow(new_array,batch_size = batch_size))
            im_array = new_array[:,:,:,0:1] 
            lb_array = new_array[:,:,:,1:2]

        yield (im_array,lb_array)
        
        
        
        
