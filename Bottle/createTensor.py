# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:04:08 2017

@author: rb130
"""
import tensorflow as tf
import cv2, numpy as np
import pickle
img_path = "C:/Users/rb117/Documents/work/Bottle/final_input/"
TrueLabels = set([10, 21, 32, 43, 55, 66, 85, 96, 107, 118, 130] )
def create_feature_sets_and_labels(test_size = 0.1):
    train_x = []
    train_y = []
    for x in range(1,139):
      label=0
      try:
        bgr_img = cv2.imread(img_path + "frame" + str(x) + '.jpg')
        img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        f_img = np.array(img.flatten()) 
        train_x.append(f_img)
        if x in TrueLabels:
          train_y.append(np.array( [0, 1] ) )
        else:
          train_y.append(np.array( [1, 0] ) )
            
      except BaseException as e:
        print(str(e))
        raise e
    print(len(train_x), len(train_y))    
    return train_x, train_y
    
