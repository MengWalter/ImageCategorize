#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 22:54:40 2018

@author: williammeng
"""

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

def load_Img(imgDir):
    #get all jpg in the folder
    img_path = gb.glob(imgDir + "/*.jpg") 
    imgNum = len(img_path)
    FolderName=[]
    for root,dirs,files in os.walk(TestImage):
        for name in files:
            FolderName.append(name)
    data = np.empty([imgNum,imageWidth*imageHeight],dtype="float32")
    
    print("Data processing")
    for i in range(imgNum):
        try:
            im = Image.open(imgDir + 'Test_' + str(i+1) + '.jpg')
            #feature engineering
            im = im.convert('L')
            # resize the images in the same size matrix
            im = scipy.misc.imresize(im, (imageHeight, imageHeight))
            im = invert(im)
            im = np.matrix(im,dtype='float')          
            data[i] = np.reshape(im,(1,imageWidth*imageHeight))
            
        except IOError,OSError:
            print("cannot open the img")
            pass
    
    return data





def main():
    img_dir = os.path.abspath('..')
    TestImage = img_dir + '/Test/'
    #export data into npyfile
    predata = load_Img(TestImage)
    np.save('predict.npy',predata)


if __name__ == '__main__':
    main()  

