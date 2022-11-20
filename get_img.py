# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:41:47 2022

@author: Administrator
"""
import csv



def get_imgid(csv_path, num_imgid):
    
    csv_train = csv.reader(open(csv_path))
    img_id = []
    
    for i in csv_train:
        # time.sleep(0.3)
        j = i[0]
        img_id.append(j)
    
    img_id = img_id[1:num_imgid+1]

    return img_id
