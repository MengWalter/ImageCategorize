#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:22:44 2018

@author: williammeng
"""

import numpy as np
from PIL import Image,  ImageFilter
import os
import math
import scipy.misc
from skimage.util import invert
imageWidth=100#图片宽度
imageHeight=100#图片高度

#每个点用他旁边的8个点及自己综合平均
def getPixel(image,x,y):  
    nearDots = image.getpixel((x,y))+image.getpixel((x - 1,y - 1))+image.getpixel((x - 1,y))+image.getpixel((x - 1,y + 1))
    +image.getpixel((x,y - 1))+image.getpixel((x,y + 1))+image.getpixel((x + 1,y - 1))+image.getpixel((x + 1,y))
    +image.getpixel((x + 1,y + 1))
    return nearDots/8  

# 降噪   
# Z: Integer 降噪次数   
def clearNoise(image,Z,data):  
    for i in range(Z):  
        for x in range(1,image.size[0] - 1):  
            for y in range(1,image.size[1] - 1):
                data[y-1,x-1] = getPixel(image,x,y) #图像的宽是矩阵的列数

# 旋转图像
#angle 旋转角度，可以为负数
def imageRotate(angle):
    im=im.rotate(angle)
    im.show()

'''图片缩小和保存'''
def ImageSave(im):
    # 获得图像尺寸:
    w, h = im.size
    print('Original image size: %sx%s' % (w, h))
    # 缩放到50%:
    im.thumbnail((w//2, h//2))
    print('Resize image to: %sx%s' % (w//2, h//2))
    # 把缩放后的图像用jpeg格式保存:
    im.save('thumbnail.jpg', 'jpeg')
    im.show()

'''#膨胀处理'''
def swell():
    newdata=data.copy()#不能用newdata=data，那样相当于指针
    for i in range(data.shape[0]-1):#shape相当于MATLAB中的size
        for j in range(data.shape[1]-1):#1表示抛弃边缘
            if(i==0 or j==0):
                continue
            for n in range(3):#3表示膨胀系数
                if(data[i,j+n-1]==0):#0是黑的，图中数字是白的
                    newdata[i,j]=0
                    break
                if(data[i+n-1,j]==0):
                    newdata[i,j]=0
                    break
    new_im = Image.fromarray(newdata*255)#如果矩阵中只有0,1,data要乘255
    new_im.show()

'''输出到文本'''
def toArray(data):
    f = open("out.txt", "w")  # 打开文件以便写入
    for i in data:
        #print(i,file = f )#python3 的写法
        print >> f, "%3d" % (i)#python2
    f.close()

'''腐蚀处理'''
def corrosion(data):
    newdata = data.copy()  # 不能用newdata=data，那样相当于指针
    for i in range(data.shape[0] - 8):  # shape相当于MATLAB中的size
        for j in range(data.shape[1] - 8):  # 18表示抛弃边缘
            if (i == 0 or j == 0):
                continue
            for n in range(10):  # 20表示膨胀系数
                if (data[i, j + n - 1] == 1):  # 0是黑的，图中数字是白的
                    newdata[i, j] = 1
                    break
                if (data[i + n - 1, j] == 1):
                    newdata[i, j] = 1
                    break
    new_im = Image.fromarray(newdata * 255)  # 如果矩阵中只有0,1,data要乘255
    new_im.show()
    # end腐蚀处理结束
    return new_im, newdata

'''图片切割'''
def carve(new_im, newdata):
    cut = []
    realcut = []
    width = newdata.shape[1]
    height = newdata.shape[0]
    for y in range(0, height):
        for x in range(0, width):
            if newdata[y, x] == 0:
                cut.append(y)
                break
            else:
                continue
    # 保存要切割的hang
    realcut.append(cut[0] - 1)
    for i in range(0, len(cut) - 1):
        if cut[i + 1] - cut[i] > 1:
            realcut.append(cut[i] + 1)  # 不相邻的两行插入两个空行
            realcut.append(cut[i + 1] - 1)
        else:
            continue
    realcut.append(cut[-1] + 1)  # 最后一行的下一行
    # 切割图片
    count = np.arange(0, len(realcut), 2)  # [0,2,4,6...]
    child_img_list = []
    for i in count:
        child_img = new_im.crop((0, realcut[i], width, realcut[i + 1]))
        child_img_list.append(child_img)

    # zong向切割
    cut_second = []
    final_img_list = []
    for i in range(0, len(child_img_list)):
        cut_second.clear()
        width = child_img_list[i].width
        height = child_img_list[i].height
        # 取有像素0的列
        for x in range(0, width):
            for y in range(realcut[i], realcut[i] + height):
                # print(i,x,y)
                if newdata[y, x] == 0:
                    cut_second.append(x)
                    break
                else:
                    continue
        # 保存要切割的lie
        realcut_second = []
        realcut_second.clear()
        realcut_second.append(cut_second[0] - 1)
        for k in range(0, len(cut_second) - 1):
            if cut_second[k + 1] - cut_second[k] > 1:
                realcut_second.append(cut_second[k] + 1)  # 不相邻的两行插入两个空行
                realcut_second.append(cut_second[k + 1] - 1)
            else:
                continue
        realcut_second.append(cut_second[-1] + 1)  # 最后一行的下一行
        # 切割图片
        count = np.arange(0, len(realcut_second), 2)  # [0,2,4,6...]
        # print(realcut_second)
        # print(count)

        for k in count:
            # print(realcut_second[k],0,realcut_second[k+1],height)
            final_img = child_img_list[i].crop((realcut_second[k], 0, realcut_second[k + 1], height))
            final_img_list.append(final_img)

            # for i in range(0,len(final_img_list)):
            # final_img_list[i].show()
            # final_img_list[i].save(str(i)+'.jpg', 'jpeg')
    return final_img_list

'''霍夫变换查找图片倾斜度'''
def method_name():
    new_im = new_im.convert('L').filter(ImageFilter.FIND_EDGES)#pillow过滤器不支持二值图，要转化成灰度图
    new_im = new_im.convert('1')
    #霍夫变换，获取直线
    newdata=new_im.getdata()
    newdata=np.matrix(newdata).reshape((new_im.height, new_im.width))#边缘检测
    ma=180#度数，每两度查找一次
    pi=3.14
    mp=math.sqrt(new_im.height*new_im.height+new_im.width*new_im.width)
    mp=math.ceil(mp)
    npp=np.zeros([ma,mp])
    for i in range(new_im.height-1):#img.height原图高度
        for j in range(new_im.width-1):#img.width 原图宽度
            if(i==0 or j==0):
                continue
            if(newdata[i,j]==255):#对边缘检测后的数据（存在newdata中）进行hough变化
                for k in range(ma):#ma=180
                    p=i*math.cos(pi*k/180)+j*math.sin(pi*k/180)#p hough变换中距离参数
                    p=math.ceil(p) #对p值优化防止为负
                    if(p!=0):
                        npp[k,p]=npp[k,p]+1  #npp对变换域中对应重复出现的点累加

    kmax=0; #最长直线的角度
    pmax=0; #最长直线的距离
    n=0; #这一部分为寻找最长直线
    yuzhi=0
    for i in range(ma): #ma=180
        for j in range(mp): #mp为原图对角线距离
            if(npp[i,j]>yuzhi): #找出最长直线 yuzhi为中间变量用于比较
                yuzhi=npp[i,j]
                kmax=i #记录最长直线的角度
    #print(kmax,yuzhi)


def main():                    
    # 打开一个jpg图像文件，注意是当前路径:
    #img_dir = " /Users/williammeng/Documents/Machine Learning/Shopee competition/Training Images "
    img_dir = os.path.abspath('..')
    BabyBibs_dir = img_dir + '/Training Images/BabyBibs/BabyBibs_300.jpg'
    im = Image.open(BabyBibs_dir)
    #imageRotate(-2)
    #ImageSave(im)
    #调整图片大小
    #im.resize((imageWidth, imageHeight),Image.ANTIALIAS) 
    im = im.convert('L')#L为灰度图，1是黑白图
    #new_im = im.filter(ImageFilter.CONTOUR)
    #new_im.show()
    im = scipy.misc.imresize(im, (imageHeight, imageHeight))
    #data = im.getdata()#变为一维矩阵
    #data = np.asarray(im,np.float32)
    #data = np.matrix(new_im,dtype='float')
    #convert matrix to image
    #imgk = Image.fromarray(data)
    #imgk.show()
    
    #data = np.reshape(data,(imageHeight, imageWidth))#变为二维矩阵
    #np.save('B_1.npy',data)
    #new_data = np.load('B_1.npy')
    im = invert(im)
    #clearNoise(im,4,data)#降噪
    #data=np.where(data > 120, 1, 0)#二值化
    #toArray(data)
    #new_im, newdata = corrosion(data)#腐蚀
    #final_img_list = carve(new_im, newdata)#图片切割
    '''矩阵转图片'''
    new_im = Image.fromarray(im)#如果矩阵中只有0,1,data要乘255
    # 显示图片
    #new_im.show()
    '''图片和标准图片进行对比（0-9,10个数字），相识度高的就是结果。'''
    scoure= [] 
    for i in range(10):
        #cut_img.clear()
        #cut_img=Image.open(str(i)+'.jpg')
        cut_img=final_img_list[i]
        for j in range(10):
            #num_img.clear()
            num_img=Image.open(str(j)+'.png')
            cut_img=cut_img.convert('1')
            num_img=num_img.convert('1')
            
            width=min(cut_img.width,num_img.width)
            height=min(cut_img.height,num_img.height)
            #print("a",width,height,cut_img.width,num_img.width,cut_img.height,num_img.height)
            cut_img=cut_img.resize((width,height))#返回值和thumbnail不一样，
            num_img=num_img.resize((width,height))
            #print(width,cut_img.width,num_img.width,height,cut_img.height,num_img.height)
            cut_img.save(str(i)+str(j)+str(0)+'.jpg', 'jpeg')
            num_img.save(str(i)+str(j)+str(1)+'.jpg', 'jpeg')
            count=0
            for m in range(0,width):
                for n in range(0,height):
                    #print(m,n,width,height,cut_img.width,num_img.width,cut_img.height,num_img.height)
                    if(cut_img.getpixel((m,n))==num_img.getpixel((m,n)) and num_img.getpixel((m,n))==0):
                        count=count+1
            scoure.append(count)  
    scoure = np.matrix(scoure).reshape((10, 10))#变为二维矩阵         
    for i in range(10):
        print(scoure[i])
    print(scoure.argmax(axis=1))

if __name__ == '__main__':
    main()  


    




