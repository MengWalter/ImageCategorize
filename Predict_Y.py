#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 08:34:29 2018

@author: williammeng
"""


import numpy as np
import csv

pre_path = 'prediction.npy'
pre_ydata = np.load(pre_path)
numy = len(pre_ydata)


with open("submit2.csv","w") as csvfile: 
    writer = csv.writer(csvfile)

    #先写入columns_name
    writer.writerow(["id","category"])
    #写入多行用writerows
    for i in range(numy):
        temp = pre_ydata[i,:]
        index = np.where(temp == max(temp))
        yvalue = index[0][0]
        #data = str(i+1) + ',' + str(yvalue)
        writer.writerow([i+1,yvalue])


    
    
    pre_path = 'predictclass.npy'
    pre_ydata = np.load(pre_path)
    numy = len(pre_ydata)
    
    with open("submit5.csv","w") as csvfile: 
        writer = csv.writer(csvfile)
    
        #先写入columns_name
        writer.writerow(["id","category"])
        #写入多行用writerows
        for i in range(numy):
            yvalue = pre_ydata[i]
            #data = str(i+1) + ',' + str(yvalue)
            writer.writerow([i+1,yvalue])