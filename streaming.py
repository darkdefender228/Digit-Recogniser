#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 17:57:12 2018

@author: romanilechko
"""
import cv2
import numpy as np
import time
from sklearn.externals import joblib
from skimage.feature import hog
from recwork import make_lines, cropping

CAM_ID = 0 # default 0
PATH_TO_MODEL = "digits_cls.pkl"
threshold = 80 # default 80

def recognize(grayscaled, im):
    clf = joblib.load(PATH_TO_MODEL)
    blured = cv2.GaussianBlur(grayscaled, (5, 5), 0)

    ret, im_th = cv2.threshold(blured, threshold, 255, cv2.THRESH_BINARY_INV)    
    res = np.where(im_th == 255)
    res_x = []
    res_y = []
    for index, r_x in enumerate(res[1]):
        if r_x > 100 and r_x < 900:
            if res[0][index] > 50 and res[0][index] < 600:
                res_x.append(r_x)
                res_y.append(res[0][index])
                
    res_x = np.asarray(res_x)
    res_y = np.asarray(res_y)
    res_y.sort()
    y = make_lines(res_y, 19, 20)
    res_x.sort()
    x = make_lines(res_x, 19, 19)
    
    coords = cropping(y, x, im_th) # coords of img y, x

    for coord in coords:
        d_obj = im_th[coord[0]:coord[2], coord[1]:coord[3]]
        
        if d_obj.shape[0] < 28 or d_obj.shape[1] < 28:
            continue
        if 255 not in d_obj:
            continue
            
        cv2.rectangle(im, (coord[1], coord[0]), (coord[3], coord[2]), (0, 255, 0), 3)
        d_obj = cv2.resize(d_obj, (28, 28), interpolation=cv2.INTER_AREA)
        d_obj = cv2.dilate(d_obj, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(d_obj, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        cv2.putText(im, str(int(nbr[0])), (coord[1], coord[0]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        cv2.imshow('frame',im)
    return im, im_th

def cam_handler(cam_id):
    cap = cv2.VideoCapture(cam_id)
    start = time.time()
    
    currentFrame = 0
    while(cap.isOpened()):
        end = time.time()
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        frame = recognize(gray, frame)
        currentFrame += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("FPS ", currentFrame/(end - start))
    cap.release()
    cv2.destroyAllWindows()

cam_handler(cam_id=CAM_ID)