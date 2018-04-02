#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:00:42 2018

@author: williammeng
"""
#open cv image pretrain
import numpy as np
from PIL import Image,  ImageFilter
import os
import math
import scipy.misc
import glob as gb
from skimage import measure            # to find shape contour
import scipy.ndimage as ndi            # to determine shape centrality
from skimage.util import invert

imageWidth=50#图片宽度
imageHeight=50#图片高度


def load_Img(imgDir,imgFoldName):
    #get all jpg in the folder
    img_path = gb.glob(imgDir+imgFoldName +"/*.jpg") 
    imgNum = len(img_path)
    
    data = np.empty([imgNum-500,imageWidth*imageHeight],dtype="float32")
    valdata = np.empty([500,imageWidth*imageHeight],dtype="float32")

    print("Data processing")
    for i in range(imgNum):
        try:
            im = Image.open(img_path[i])
            #feature engineering
            im = im.convert('L')
            # resize the images in the same size matrix
            im = scipy.misc.imresize(im, (imageHeight, imageHeight))
            im = invert(im)
            im = np.matrix(im,dtype='float')
            if i < imgNum -500:
                data[i] = np.reshape(im,(1,imageWidth*imageHeight))
            else:
                valdata[i-(imgNum -500)] = np.reshape(im,(1,imageWidth*imageHeight))
        except IOError,OSError:
            print("cannot open the img")
            pass
    
    return data,valdata



def main():
    img_dir = os.path.abspath('..')
    TrainingImage = img_dir + '/Training Images/'
    
    #Get folder name
    FolderName=[]
    for root,dirs,files in os.walk(TrainingImage):
        for name in dirs:
            FolderName.append(name)
    
    #export data into npyfile
    # check if the document exist
    isexist1 = os.path.exists('Training data')
    if isexist1==False:
        os.makedirs('Training data')
    isexist2 = os.path.exists('Validate data')
    if isexist2==False:
        os.makedirs('Validate data')
    
    for i in range(len(FolderName)):
        data,valdata = load_Img(TrainingImage,FolderName[i])
        np.save('Training data/' + FolderName[i]+'.npy',data)
        np.save('Validate data/' + FolderName[i]+'.npy',valdata)


if __name__ == '__main__':
    main()  



