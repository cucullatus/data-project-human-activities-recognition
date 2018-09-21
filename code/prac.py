# -*- coding: utf-8 -*-

import math
import scipy.stats as st
import numpy as np
from sklearn.model_selection import train_test_split
import cPickle as cp
import math
from math import sqrt
from sklearn import preprocessing
from hmmlearn.base import _BaseHMM 
from hmmlearn import hmm
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import GMMHMM
import matplotlib.pyplot as plt
import random
import types
from generateData import *
from gmm3 import *
import ch
ch.set_ch()
import matplotlib.pyplot as plt
import matplotlib as mpl


def multipl(a,b):             #相关系数两个def
    sumofab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sumofab+=temp
    return sumofab
 
def corrcoef(x,y):
    n=len(x)
    #求和
    sum1=sum(x)
    sum2=sum(y)
    #求乘积之和
    sumofxy=multipl(x,y)
    #求平方和
    sumofx2 = sum([pow(i,2) for i in x])
    sumofy2 = sum([pow(j,2) for j in y])
    num=sumofxy-(float(sum1)*float(sum2)/n)
    #计算皮尔逊相关系数
    den=sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))
    return num/den
#算skew,kurt	
def calc(data):
    n = len(data)
    niu = 0.0
    niu2 = 0.0
    niu3 = 0.0
    for a in data:
        niu += a
        niu2 += a**2
        niu3 += a**3
    niu/= n   #这是求E(X)
    niu2 /= n #这是E(X^2)
    niu3 /= n #这是E(X^3)
    sigma = math.sqrt(niu2 - niu*niu) #这是D（X）的开方，标准差
    return [niu,sigma,niu3] #返回[E（X）,标准差，E（X^3）]

def calc_stat(data):
    [niu,sigma,niu3] = calc(data)
    n = len(data)
    niu4 = 0.0
    for a in data:
        a -= niu
        niu4 += a ** 4
    niu4 /= n   
    skew = (niu3 - 3*niu*sigma**2 - niu**3)/(sigma**3)
    kurt =  niu4/(sigma**2)
   
    return skew,kurt #返回了偏度，峰度

def autocorrelation(x,lags):#计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
    n = len(x)
    
    result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]/(x[i:].std()*x[:n-i].std()*(n-i)) 
          for i in range(1,lags+1)]
    return result

ac_label=(0,1,2,3,4)

f=open(r"C:\test\data05.pkl")        #加索引操作
list_5=cp.load(f)                    #共有5*8*60*125行，45列
seg_data=[]
for line in list_5:
    seg_data.append(line)            #{array(),array(), .....,}格式
gravity_data=[]
for index,d in enumerate(seg_data):   #贴标签，取组合加速度值gravity，，，，训练数据改这里
    T_acc_x=d[0]                       #五肢加速度信号
    T_acc_y=d[1]
    T_acc_z=d[2]
	
    RA_acc_x=d[9]
    RA_acc_y=d[10]
    RA_acc_z=d[11]
	
    LA_acc_x=d[18]
    LA_acc_y=d[19]
    LA_acc_z=d[20]
	
    RL_acc_x=d[27]
    RL_acc_y=d[28]
    RL_acc_z=d[29]

    LL_acc_x=d[36]
    LL_acc_y=d[37]
    LL_acc_z=d[38]	
	
    x=8*60*125          #标签操作
    if index<=x-1:
        label=0
    elif index<=2*x-1:
        label=1
    elif index<=3*x-1:
        label=2
    elif index<=4*x-1:
        label=3
    else:
        label=4
		
    #求组合三轴加速度值
#    T_gravity=math.sqrt(math.pow(T_acc_x,2)+math.pow(T_acc_y,2)+math.pow(T_acc_z,2))
#    RA_gravity=math.sqrt(math.pow(RA_acc_x,2)+math.pow(RA_acc_y,2)+math.pow(RA_acc_z,2))
#    LA_gravity=math.sqrt(math.pow(LA_acc_x,2)+math.pow(LA_acc_y,2)+math.pow(LA_acc_z,2))
#    RL_gravity=math.sqrt(math.pow(RL_acc_x,2)+math.pow(RL_acc_y,2)+math.pow(RL_acc_z,2))
#    LL_gravity=math.sqrt(math.pow(LL_acc_x,2)+math.pow(LL_acc_y,2)+math.pow(LL_acc_z,2))
	
	
    gravity_dict={"T_acc_x":T_acc_x,"T_acc_y":T_acc_y,"T_acc_z":T_acc_z,"RA_acc_x":RA_acc_x,"RA_acc_y":RA_acc_y,"RA_acc_z":RA_acc_z,"LA_acc_x":LA_acc_x,"LA_acc_y":LA_acc_y,"LA_acc_z":LA_acc_z,"RL_acc_x":RL_acc_x,"RL_acc_y":RL_acc_y,"RL_acc_z":RL_acc_z,"LL_acc_x":LL_acc_x,"LL_acc_y":LL_acc_y,"LL_acc_z":LL_acc_z,"label":label} #无序，但可通过key 取value
    gravity_data.append(gravity_dict)



#分离5个gravity数据
split_data=[]
T_acc_x=[]                       #五肢加速度信号
T_acc_y=[]
T_acc_z=[]
	
RA_acc_x=[]
RA_acc_y=[]
RA_acc_z=[]
	
LA_acc_x=[]
LA_acc_y=[]
LA_acc_z=[]
	
RL_acc_x=[]
RL_acc_y=[]
RL_acc_z=[]

LL_acc_x=[]
LL_acc_y=[]
LL_acc_z=[]	
counter=0     #计算8*60*125列
last_label=gravity_data[0]["label"]
for gravity_dict in gravity_data:    #遍历通过keys取values
    T_acc_x.append(gravity_dict["T_acc_x"])           #注意for循环的顺序，尤其是最后一个的位置
    T_acc_y.append(gravity_dict["T_acc_y"])
    T_acc_z.append(gravity_dict["T_acc_z"])
    RA_acc_x.append(gravity_dict["RA_acc_x"])
    RA_acc_y.append(gravity_dict["RA_acc_y"])
    RA_acc_z.append(gravity_dict["RA_acc_z"])
    LA_acc_x.append(gravity_dict["LA_acc_x"])           #注意for循环的顺序，尤其是最后一个的位置
    LA_acc_y.append(gravity_dict["LA_acc_y"])
    LA_acc_z.append(gravity_dict["LA_acc_z"])
    RL_acc_x.append(gravity_dict["RL_acc_x"])
    RL_acc_y.append(gravity_dict["RL_acc_y"])
    RL_acc_z.append(gravity_dict["RL_acc_z"])
    LL_acc_x.append(gravity_dict["LL_acc_x"])
    LL_acc_y.append(gravity_dict["LL_acc_y"])
    LL_acc_z.append(gravity_dict["LL_acc_z"])
	
    last_label=gravity_dict["label"]
    counter+=1
    if not(counter<125 and gravity_dict["label"]==last_label):#counter<480*125值改成125，480*5类
        seg_data={"label":last_label,"T_acc_x":T_acc_x,"T_acc_y":T_acc_y,"T_acc_z":T_acc_z,"RA_acc_x":RA_acc_x,"RA_acc_y":RA_acc_y,"RA_acc_z":RA_acc_z,"LA_acc_x":LA_acc_x,"LA_acc_y":LA_acc_y,"LA_acc_z":LA_acc_z,"RL_acc_x":RL_acc_x,"RL_acc_y":RL_acc_y,"RL_acc_z":RL_acc_z,"LL_acc_x":LL_acc_x,"LL_acc_y":LL_acc_y,"LL_acc_z":LL_acc_z}
#        print seg_data
        split_data.append(seg_data)
        T_acc_x=[]                       #五肢加速度信号
        T_acc_y=[]
        T_acc_z=[]
	
        RA_acc_x=[]
        RA_acc_y=[]
        RA_acc_z=[]
	
        LA_acc_x=[]
        LA_acc_y=[]
        LA_acc_z=[]
	
        RL_acc_x=[]
        RL_acc_y=[]
        RL_acc_z=[]

        LL_acc_x=[]
        LL_acc_y=[]
        LL_acc_z=[]	
        counter=0 
#print split_data    #len=2400
#计算5*3特征值
statistics_data=[]
Collect=[]              #整合list，标准化
labels=[]               #放标签

init_T_acc_x=split_data[0]["T_acc_x"]  #相关系数初始化，使index齐全
init_T_acc_y=split_data[0]["T_acc_y"]
init_T_acc_z=split_data[0]["T_acc_z"]
init_RA_acc_x=split_data[0]["RA_acc_x"]
init_RA_acc_y=split_data[0]["RA_acc_y"]
init_RA_acc_z=split_data[0]["RA_acc_z"]
init_LA_acc_x=split_data[0]["LA_acc_x"]  #相关系数初始化，使index齐全
init_LA_acc_y=split_data[0]["LA_acc_y"]
init_LA_acc_z=split_data[0]["LA_acc_z"]
init_RL_acc_x=split_data[0]["RL_acc_x"]
init_RL_acc_y=split_data[0]["RL_acc_y"]
init_RL_acc_z=split_data[0]["RL_acc_z"]
init_LL_acc_x=split_data[0]["LL_acc_x"]
init_LL_acc_y=split_data[0]["LL_acc_y"]
init_LL_acc_z=split_data[0]["LL_acc_z"]
for index,seg_data in enumerate(split_data):
    T_acc_x=np.array(seg_data.pop("T_acc_x")) 
    seg_data["T_acc_x_mean"]=np.mean(T_acc_x)
    Collect.append(seg_data["T_acc_x_mean"])
	
    seg_data["T_acc_x_var"]=np.var(T_acc_x)
    Collect.append(seg_data["T_acc_x_var"])
	
    seg_data["T_acc_x_ptp"]=np.ptp(T_acc_x)
    Collect.append(seg_data["T_acc_x_ptp"])
    
    if index<=2398:
        seg_data["T_acc_x_corrcoef"]=corrcoef(T_acc_x,np.array(split_data[index+1]["T_acc_x"]))
        Collect.append(seg_data["T_acc_x_corrcoef"])
    else:
        seg_data["T_acc_x_corrcoef"]=corrcoef(T_acc_x,np.array(init_T_acc_x))
        Collect.append(seg_data["T_acc_x_corrcoef"])
    skew,kurt=calc_stat(T_acc_x)
    seg_data["T_acc_x_skew"]=skew
    Collect.append(seg_data["T_acc_x_skew"])
    seg_data["T_acc_x_kurt"]=kurt
    Collect.append(seg_data["T_acc_x_kurt"])
    
    peak=np.argsort(-np.fft.fft(T_acc_x))
    seg_data["T_acc_x_peak1"]=peak[0]
    Collect.append(seg_data["T_acc_x_peak1"])
    
    seg_data["T_acc_x_peak2"]=peak[1]
    Collect.append(seg_data["T_acc_x_peak2"])
	
    seg_data["T_acc_x_peak3"]=peak[2]
    Collect.append(seg_data["T_acc_x_peak3"])	

    seg_data["T_acc_x_peak4"]=peak[3]
    Collect.append(seg_data["T_acc_x_peak4"])
	
    seg_data["T_acc_x_peak5"]=peak[4]
    Collect.append(seg_data["T_acc_x_peak5"])
	
    autocor=autocorrelation(T_acc_x,12)
    seg_data["T_acc_x_autocor0"]=autocor[0]
    Collect.append(seg_data["T_acc_x_autocor0"])
    
    seg_data["T_acc_x_autocor1"]=autocor[1]
    Collect.append(seg_data["T_acc_x_autocor1"])
	
    seg_data["T_acc_x_autocor2"]=autocor[2]
    Collect.append(seg_data["T_acc_x_autocor2"])
    
    seg_data["T_acc_x_autocor3"]=autocor[3]
    Collect.append(seg_data["T_acc_x_autocor3"])
	
    seg_data["T_acc_x_autocor4"]=autocor[4]
    Collect.append(seg_data["T_acc_x_autocor4"])
    
    seg_data["T_acc_x_autocor5"]=autocor[5]
    Collect.append(seg_data["T_acc_x_autocor5"])
	
    seg_data["T_acc_x_autocor6"]=autocor[6]
    Collect.append(seg_data["T_acc_x_autocor6"])
    
    seg_data["T_acc_x_autocor7"]=autocor[7]
    Collect.append(seg_data["T_acc_x_autocor7"])
	
    seg_data["T_acc_x_autocor8"]=autocor[8]
    Collect.append(seg_data["T_acc_x_autocor8"])
    
    seg_data["T_acc_x_autocor9"]=autocor[9]
    Collect.append(seg_data["T_acc_x_autocor9"])
	
    T_acc_y=np.array(seg_data.pop("T_acc_y")) 
    seg_data["T_acc_y_mean"]=np.mean(T_acc_y)
    Collect.append(seg_data["T_acc_y_mean"])
	
    seg_data["T_acc_y_var"]=np.var(T_acc_y)
    Collect.append(seg_data["T_acc_y_var"])
	
    seg_data["T_acc_y_ptp"]=np.ptp(T_acc_y)
    Collect.append(seg_data["T_acc_y_ptp"])
    
    if index<=2398:
        seg_data["T_acc_y_corrcoef"]=corrcoef(T_acc_y,np.array(split_data[index+1]["T_acc_y"]))
        Collect.append(seg_data["T_acc_y_corrcoef"])
    else:
        seg_data["T_acc_y_corrcoef"]=corrcoef(T_acc_y,np.array(init_T_acc_y))
        Collect.append(seg_data["T_acc_y_corrcoef"])
    skew,kurt=calc_stat(T_acc_y)
    seg_data["T_acc_y_skew"]=skew
    Collect.append(seg_data["T_acc_y_skew"])
    seg_data["T_acc_y_kurt"]=kurt
    Collect.append(seg_data["T_acc_y_kurt"])
    
    peak=np.argsort(-np.fft.fft(T_acc_y))
    seg_data["T_acc_y_peak1"]=peak[0]
    Collect.append(seg_data["T_acc_y_peak1"])
    
    seg_data["T_acc_y_peak2"]=peak[1]
    Collect.append(seg_data["T_acc_y_peak2"])
	
    seg_data["T_acc_y_peak3"]=peak[2]
    Collect.append(seg_data["T_acc_y_peak3"])	

    seg_data["T_acc_y_peak4"]=peak[3]
    Collect.append(seg_data["T_acc_y_peak4"])
	
    seg_data["T_acc_y_peak5"]=peak[4]
    Collect.append(seg_data["T_acc_y_peak5"])
	
    autocor=autocorrelation(T_acc_y,12)
    seg_data["T_acc_y_autocor0"]=autocor[0]
    Collect.append(seg_data["T_acc_y_autocor0"])
    
    seg_data["T_acc_y_autocor1"]=autocor[1]
    Collect.append(seg_data["T_acc_y_autocor1"])
	
    seg_data["T_acc_y_autocor2"]=autocor[2]
    Collect.append(seg_data["T_acc_y_autocor2"])
    
    seg_data["T_acc_y_autocor3"]=autocor[3]
    Collect.append(seg_data["T_acc_y_autocor3"])
	
    seg_data["T_acc_y_autocor4"]=autocor[4]
    Collect.append(seg_data["T_acc_y_autocor4"])
    
    seg_data["T_acc_y_autocor5"]=autocor[5]
    Collect.append(seg_data["T_acc_y_autocor5"])
	
    seg_data["T_acc_y_autocor6"]=autocor[6]
    Collect.append(seg_data["T_acc_y_autocor6"])
    
    seg_data["T_acc_y_autocor7"]=autocor[7]
    Collect.append(seg_data["T_acc_y_autocor7"])
	
    seg_data["T_acc_y_autocor8"]=autocor[8]
    Collect.append(seg_data["T_acc_y_autocor8"])
    
    seg_data["T_acc_y_autocor9"]=autocor[9]
    Collect.append(seg_data["T_acc_y_autocor9"])
	
    T_acc_z=np.array(seg_data.pop("T_acc_z")) 
    seg_data["T_acc_z_mean"]=np.mean(T_acc_z)
    Collect.append(seg_data["T_acc_z_mean"])
	
    seg_data["T_acc_z_var"]=np.var(T_acc_z)
    Collect.append(seg_data["T_acc_z_var"])
	
    seg_data["T_acc_z_ptp"]=np.ptp(T_acc_z)
    Collect.append(seg_data["T_acc_z_ptp"])
    
    if index<=2398:
        seg_data["T_acc_z_corrcoef"]=corrcoef(T_acc_z,np.array(split_data[index+1]["T_acc_z"]))
        Collect.append(seg_data["T_acc_z_corrcoef"])
    else:
        seg_data["T_acc_z_corrcoef"]=corrcoef(T_acc_z,np.array(init_T_acc_z))
        Collect.append(seg_data["T_acc_z_corrcoef"])
    skew,kurt=calc_stat(T_acc_z)
    seg_data["T_acc_z_skew"]=skew
    Collect.append(seg_data["T_acc_z_skew"])
    seg_data["T_acc_z_kurt"]=kurt
    Collect.append(seg_data["T_acc_z_kurt"])
    
    peak=np.argsort(-np.fft.fft(T_acc_z))
    seg_data["T_acc_z_peak1"]=peak[0]
    Collect.append(seg_data["T_acc_z_peak1"])
    
    seg_data["T_acc_z_peak2"]=peak[1]
    Collect.append(seg_data["T_acc_z_peak2"])
	
    seg_data["T_acc_z_peak3"]=peak[2]
    Collect.append(seg_data["T_acc_z_peak3"])	

    seg_data["T_acc_z_peak4"]=peak[3]
    Collect.append(seg_data["T_acc_z_peak4"])
	
    seg_data["T_acc_z_peak5"]=peak[4]
    Collect.append(seg_data["T_acc_z_peak5"])
	
    autocor=autocorrelation(T_acc_z,12)
    seg_data["T_acc_z_autocor0"]=autocor[0]
    Collect.append(seg_data["T_acc_z_autocor0"])
    
    seg_data["T_acc_z_autocor1"]=autocor[1]
    Collect.append(seg_data["T_acc_z_autocor1"])
	
    seg_data["T_acc_z_autocor2"]=autocor[2]
    Collect.append(seg_data["T_acc_z_autocor2"])
    
    seg_data["T_acc_z_autocor3"]=autocor[3]
    Collect.append(seg_data["T_acc_z_autocor3"])
	
    seg_data["T_acc_z_autocor4"]=autocor[4]
    Collect.append(seg_data["T_acc_z_autocor4"])
    
    seg_data["T_acc_z_autocor5"]=autocor[5]
    Collect.append(seg_data["T_acc_z_autocor5"])
	
    seg_data["T_acc_z_autocor6"]=autocor[6]
    Collect.append(seg_data["T_acc_z_autocor6"])
    
    seg_data["T_acc_z_autocor7"]=autocor[7]
    Collect.append(seg_data["T_acc_z_autocor7"])
	
    seg_data["T_acc_z_autocor8"]=autocor[8]
    Collect.append(seg_data["T_acc_z_autocor8"])
    
    seg_data["T_acc_z_autocor9"]=autocor[9]
    Collect.append(seg_data["T_acc_z_autocor9"])
	
    RA_acc_x=np.array(seg_data.pop("RA_acc_x")) 
    seg_data["RA_acc_x_mean"]=np.mean(RA_acc_x)
    Collect.append(seg_data["RA_acc_x_mean"])
	
    seg_data["RA_acc_x_var"]=np.var(RA_acc_x)
    Collect.append(seg_data["RA_acc_x_var"])
	
    seg_data["RA_acc_x_ptp"]=np.ptp(RA_acc_x)
    Collect.append(seg_data["RA_acc_x_ptp"])
    
    if index<=2398:
        seg_data["RA_acc_x_corrcoef"]=corrcoef(RA_acc_x,np.array(split_data[index+1]["RA_acc_x"]))
        Collect.append(seg_data["RA_acc_x_corrcoef"])
    else:
        seg_data["RA_acc_x_corrcoef"]=corrcoef(RA_acc_x,np.array(init_RA_acc_x))
        Collect.append(seg_data["RA_acc_x_corrcoef"])
    skew,kurt=calc_stat(RA_acc_x)
    seg_data["RA_acc_x_skew"]=skew
    Collect.append(seg_data["RA_acc_x_skew"])
    seg_data["RA_acc_x_kurt"]=kurt
    Collect.append(seg_data["RA_acc_x_kurt"])
    
    peak=np.argsort(-np.fft.fft(RA_acc_x))
    seg_data["RA_acc_x_peak1"]=peak[0]
    Collect.append(seg_data["RA_acc_x_peak1"])
    
    seg_data["RA_acc_x_peak2"]=peak[1]
    Collect.append(seg_data["RA_acc_x_peak2"])
	
    seg_data["RA_acc_x_peak3"]=peak[2]
    Collect.append(seg_data["RA_acc_x_peak3"])	

    seg_data["RA_acc_x_peak4"]=peak[3]
    Collect.append(seg_data["RA_acc_x_peak4"])
	
    seg_data["RA_acc_x_peak5"]=peak[4]
    Collect.append(seg_data["RA_acc_x_peak5"])
	
    autocor=autocorrelation(RA_acc_x,12)
    seg_data["RA_acc_x_autocor0"]=autocor[0]
    Collect.append(seg_data["RA_acc_x_autocor0"])
    
    seg_data["RA_acc_x_autocor1"]=autocor[1]
    Collect.append(seg_data["RA_acc_x_autocor1"])
	
    seg_data["RA_acc_x_autocor2"]=autocor[2]
    Collect.append(seg_data["RA_acc_x_autocor2"])
    
    seg_data["RA_acc_x_autocor3"]=autocor[3]
    Collect.append(seg_data["RA_acc_x_autocor3"])
	
    seg_data["RA_acc_x_autocor4"]=autocor[4]
    Collect.append(seg_data["RA_acc_x_autocor4"])
    
    seg_data["RA_acc_x_autocor5"]=autocor[5]
    Collect.append(seg_data["RA_acc_x_autocor5"])
	
    seg_data["RA_acc_x_autocor6"]=autocor[6]
    Collect.append(seg_data["RA_acc_x_autocor6"])
    
    seg_data["RA_acc_x_autocor7"]=autocor[7]
    Collect.append(seg_data["RA_acc_x_autocor7"])
	
    seg_data["RA_acc_x_autocor8"]=autocor[8]
    Collect.append(seg_data["RA_acc_x_autocor8"])
    
    seg_data["RA_acc_x_autocor9"]=autocor[9]
    Collect.append(seg_data["RA_acc_x_autocor9"])
	
    RA_acc_y=np.array(seg_data.pop("RA_acc_y")) 
    seg_data["RA_acc_y_mean"]=np.mean(RA_acc_y)
    Collect.append(seg_data["RA_acc_y_mean"])
	
    seg_data["RA_acc_y_var"]=np.var(RA_acc_y)
    Collect.append(seg_data["RA_acc_y_var"])
	
    seg_data["RA_acc_y_ptp"]=np.ptp(RA_acc_y)
    Collect.append(seg_data["RA_acc_y_ptp"])
    
    if index<=2398:
        seg_data["RA_acc_y_corrcoef"]=corrcoef(RA_acc_y,np.array(split_data[index+1]["RA_acc_y"]))
        Collect.append(seg_data["RA_acc_y_corrcoef"])
    else:
        seg_data["RA_acc_y_corrcoef"]=corrcoef(RA_acc_y,np.array(init_RA_acc_y))
        Collect.append(seg_data["RA_acc_y_corrcoef"])
    skew,kurt=calc_stat(RA_acc_y)
    seg_data["RA_acc_y_skew"]=skew
    Collect.append(seg_data["RA_acc_y_skew"])
    seg_data["RA_acc_y_kurt"]=kurt
    Collect.append(seg_data["RA_acc_y_kurt"])
    
    peak=np.argsort(-np.fft.fft(RA_acc_y))
    seg_data["RA_acc_y_peak1"]=peak[0]
    Collect.append(seg_data["RA_acc_y_peak1"])
    
    seg_data["RA_acc_y_peak2"]=peak[1]
    Collect.append(seg_data["RA_acc_y_peak2"])
	
    seg_data["RA_acc_y_peak3"]=peak[2]
    Collect.append(seg_data["RA_acc_y_peak3"])	

    seg_data["RA_acc_y_peak4"]=peak[3]
    Collect.append(seg_data["RA_acc_y_peak4"])
	
    seg_data["RA_acc_y_peak5"]=peak[4]
    Collect.append(seg_data["RA_acc_y_peak5"])
	
    autocor=autocorrelation(RA_acc_y,12)
    seg_data["RA_acc_y_autocor0"]=autocor[0]
    Collect.append(seg_data["RA_acc_y_autocor0"])
    
    seg_data["RA_acc_y_autocor1"]=autocor[1]
    Collect.append(seg_data["RA_acc_y_autocor1"])
	
    seg_data["RA_acc_y_autocor2"]=autocor[2]
    Collect.append(seg_data["RA_acc_y_autocor2"])
    
    seg_data["RA_acc_y_autocor3"]=autocor[3]
    Collect.append(seg_data["RA_acc_y_autocor3"])
	
    seg_data["RA_acc_y_autocor4"]=autocor[4]
    Collect.append(seg_data["RA_acc_y_autocor4"])
    
    seg_data["RA_acc_y_autocor5"]=autocor[5]
    Collect.append(seg_data["RA_acc_y_autocor5"])
	
    seg_data["RA_acc_y_autocor6"]=autocor[6]
    Collect.append(seg_data["RA_acc_y_autocor6"])
    
    seg_data["RA_acc_y_autocor7"]=autocor[7]
    Collect.append(seg_data["RA_acc_y_autocor7"])
	
    seg_data["RA_acc_y_autocor8"]=autocor[8]
    Collect.append(seg_data["RA_acc_y_autocor8"])
    
    seg_data["RA_acc_y_autocor9"]=autocor[9]
    Collect.append(seg_data["RA_acc_y_autocor9"])
	
    RA_acc_z=np.array(seg_data.pop("RA_acc_z")) 
    seg_data["RA_acc_z_mean"]=np.mean(RA_acc_z)
    Collect.append(seg_data["RA_acc_z_mean"])
	
    seg_data["RA_acc_z_var"]=np.var(RA_acc_z)
    Collect.append(seg_data["RA_acc_z_var"])
	
    seg_data["RA_acc_z_ptp"]=np.ptp(RA_acc_z)
    Collect.append(seg_data["RA_acc_z_ptp"])
    
    if index<=2398:
        seg_data["RA_acc_z_corrcoef"]=corrcoef(RA_acc_z,np.array(split_data[index+1]["RA_acc_z"]))
        Collect.append(seg_data["RA_acc_z_corrcoef"])
    else:
        seg_data["RA_acc_z_corrcoef"]=corrcoef(RA_acc_z,np.array(init_RA_acc_z))
        Collect.append(seg_data["RA_acc_z_corrcoef"])
    skew,kurt=calc_stat(RA_acc_z)
    seg_data["RA_acc_z_skew"]=skew
    Collect.append(seg_data["RA_acc_z_skew"])
    seg_data["RA_acc_z_kurt"]=kurt
    Collect.append(seg_data["RA_acc_z_kurt"])
    
    peak=np.argsort(-np.fft.fft(RA_acc_z))
    seg_data["RA_acc_z_peak1"]=peak[0]
    Collect.append(seg_data["RA_acc_z_peak1"])
    
    seg_data["RA_acc_z_peak2"]=peak[1]
    Collect.append(seg_data["RA_acc_z_peak2"])
	
    seg_data["RA_acc_z_peak3"]=peak[2]
    Collect.append(seg_data["RA_acc_z_peak3"])	

    seg_data["RA_acc_z_peak4"]=peak[3]
    Collect.append(seg_data["RA_acc_z_peak4"])
	
    seg_data["RA_acc_z_peak5"]=peak[4]
    Collect.append(seg_data["RA_acc_z_peak5"])
	
    autocor=autocorrelation(RA_acc_z,12)
    seg_data["RA_acc_z_autocor0"]=autocor[0]
    Collect.append(seg_data["RA_acc_z_autocor0"])
    
    seg_data["RA_acc_z_autocor1"]=autocor[1]
    Collect.append(seg_data["RA_acc_z_autocor1"])
	
    seg_data["RA_acc_z_autocor2"]=autocor[2]
    Collect.append(seg_data["RA_acc_z_autocor2"])
    
    seg_data["RA_acc_z_autocor3"]=autocor[3]
    Collect.append(seg_data["RA_acc_z_autocor3"])
	
    seg_data["RA_acc_z_autocor4"]=autocor[4]
    Collect.append(seg_data["RA_acc_z_autocor4"])
    
    seg_data["RA_acc_z_autocor5"]=autocor[5]
    Collect.append(seg_data["RA_acc_z_autocor5"])
	
    seg_data["RA_acc_z_autocor6"]=autocor[6]
    Collect.append(seg_data["RA_acc_z_autocor6"])
    
    seg_data["RA_acc_z_autocor7"]=autocor[7]
    Collect.append(seg_data["RA_acc_z_autocor7"])
	
    seg_data["RA_acc_z_autocor8"]=autocor[8]
    Collect.append(seg_data["RA_acc_z_autocor8"])
    
    seg_data["RA_acc_z_autocor9"]=autocor[9]
    Collect.append(seg_data["RA_acc_z_autocor9"])
	
    LA_acc_x=np.array(seg_data.pop("LA_acc_x")) 
    seg_data["LA_acc_x_mean"]=np.mean(LA_acc_x)
    Collect.append(seg_data["LA_acc_x_mean"])
	
    seg_data["LA_acc_x_var"]=np.var(LA_acc_x)
    Collect.append(seg_data["LA_acc_x_var"])
	
    seg_data["LA_acc_x_ptp"]=np.ptp(LA_acc_x)
    Collect.append(seg_data["LA_acc_x_ptp"])
    
    if index<=2398:
        seg_data["LA_acc_x_corrcoef"]=corrcoef(LA_acc_x,np.array(split_data[index+1]["LA_acc_x"]))
        Collect.append(seg_data["LA_acc_x_corrcoef"])
    else:
        seg_data["LA_acc_x_corrcoef"]=corrcoef(LA_acc_x,np.array(init_LA_acc_x))
        Collect.append(seg_data["LA_acc_x_corrcoef"])
    skew,kurt=calc_stat(LA_acc_x)
    seg_data["LA_acc_x_skew"]=skew
    Collect.append(seg_data["LA_acc_x_skew"])
    seg_data["LA_acc_x_kurt"]=kurt
    Collect.append(seg_data["LA_acc_x_kurt"])
    
    peak=np.argsort(-np.fft.fft(LA_acc_x))
    seg_data["LA_acc_x_peak1"]=peak[0]
    Collect.append(seg_data["LA_acc_x_peak1"])
    
    seg_data["LA_acc_x_peak2"]=peak[1]
    Collect.append(seg_data["LA_acc_x_peak2"])
	
    seg_data["LA_acc_x_peak3"]=peak[2]
    Collect.append(seg_data["LA_acc_x_peak3"])	

    seg_data["LA_acc_x_peak4"]=peak[3]
    Collect.append(seg_data["LA_acc_x_peak4"])
	
    seg_data["LA_acc_x_peak5"]=peak[4]
    Collect.append(seg_data["LA_acc_x_peak5"])
	
    autocor=autocorrelation(LA_acc_x,12)
    seg_data["LA_acc_x_autocor0"]=autocor[0]
    Collect.append(seg_data["LA_acc_x_autocor0"])
    
    seg_data["LA_acc_x_autocor1"]=autocor[1]
    Collect.append(seg_data["LA_acc_x_autocor1"])
	
    seg_data["LA_acc_x_autocor2"]=autocor[2]
    Collect.append(seg_data["LA_acc_x_autocor2"])
    
    seg_data["LA_acc_x_autocor3"]=autocor[3]
    Collect.append(seg_data["LA_acc_x_autocor3"])
	
    seg_data["LA_acc_x_autocor4"]=autocor[4]
    Collect.append(seg_data["LA_acc_x_autocor4"])
    
    seg_data["LA_acc_x_autocor5"]=autocor[5]
    Collect.append(seg_data["LA_acc_x_autocor5"])
	
    seg_data["LA_acc_x_autocor6"]=autocor[6]
    Collect.append(seg_data["LA_acc_x_autocor6"])
    
    seg_data["LA_acc_x_autocor7"]=autocor[7]
    Collect.append(seg_data["LA_acc_x_autocor7"])
	
    seg_data["LA_acc_x_autocor8"]=autocor[8]
    Collect.append(seg_data["LA_acc_x_autocor8"])
    
    seg_data["LA_acc_x_autocor9"]=autocor[9]
    Collect.append(seg_data["LA_acc_x_autocor9"])
	
    LA_acc_y=np.array(seg_data.pop("LA_acc_y")) 
    seg_data["LA_acc_y_mean"]=np.mean(LA_acc_y)
    Collect.append(seg_data["LA_acc_y_mean"])
	
    seg_data["LA_acc_y_var"]=np.var(LA_acc_y)
    Collect.append(seg_data["LA_acc_y_var"])
	
    seg_data["LA_acc_y_ptp"]=np.ptp(LA_acc_y)
    Collect.append(seg_data["LA_acc_y_ptp"])
    
    if index<=2398:
        seg_data["LA_acc_y_corrcoef"]=corrcoef(LA_acc_y,np.array(split_data[index+1]["LA_acc_y"]))
        Collect.append(seg_data["LA_acc_y_corrcoef"])
    else:
        seg_data["LA_acc_y_corrcoef"]=corrcoef(LA_acc_y,np.array(init_LA_acc_y))
        Collect.append(seg_data["LA_acc_y_corrcoef"])
    skew,kurt=calc_stat(LA_acc_y)
    seg_data["LA_acc_y_skew"]=skew
    Collect.append(seg_data["LA_acc_y_skew"])
    seg_data["LA_acc_y_kurt"]=kurt
    Collect.append(seg_data["LA_acc_y_kurt"])
    
    peak=np.argsort(-np.fft.fft(LA_acc_y))
    seg_data["LA_acc_y_peak1"]=peak[0]
    Collect.append(seg_data["LA_acc_y_peak1"])
    
    seg_data["LA_acc_y_peak2"]=peak[1]
    Collect.append(seg_data["LA_acc_y_peak2"])
	
    seg_data["LA_acc_y_peak3"]=peak[2]
    Collect.append(seg_data["LA_acc_y_peak3"])	

    seg_data["LA_acc_y_peak4"]=peak[3]
    Collect.append(seg_data["LA_acc_y_peak4"])
	
    seg_data["LA_acc_y_peak5"]=peak[4]
    Collect.append(seg_data["LA_acc_y_peak5"])
	
    autocor=autocorrelation(LA_acc_y,12)
    seg_data["LA_acc_y_autocor0"]=autocor[0]
    Collect.append(seg_data["LA_acc_y_autocor0"])
    
    seg_data["LA_acc_y_autocor1"]=autocor[1]
    Collect.append(seg_data["LA_acc_y_autocor1"])
	
    seg_data["LA_acc_y_autocor2"]=autocor[2]
    Collect.append(seg_data["LA_acc_y_autocor2"])
    
    seg_data["LA_acc_y_autocor3"]=autocor[3]
    Collect.append(seg_data["LA_acc_y_autocor3"])
	
    seg_data["LA_acc_y_autocor4"]=autocor[4]
    Collect.append(seg_data["LA_acc_y_autocor4"])
    
    seg_data["LA_acc_y_autocor5"]=autocor[5]
    Collect.append(seg_data["LA_acc_y_autocor5"])
	
    seg_data["LA_acc_y_autocor6"]=autocor[6]
    Collect.append(seg_data["LA_acc_y_autocor6"])
    
    seg_data["LA_acc_y_autocor7"]=autocor[7]
    Collect.append(seg_data["LA_acc_y_autocor7"])
	
    seg_data["LA_acc_y_autocor8"]=autocor[8]
    Collect.append(seg_data["LA_acc_y_autocor8"])
    
    seg_data["LA_acc_y_autocor9"]=autocor[9]
    Collect.append(seg_data["LA_acc_y_autocor9"])
	
    LA_acc_z=np.array(seg_data.pop("LA_acc_z")) 
    seg_data["LA_acc_z_mean"]=np.mean(LA_acc_z)
    Collect.append(seg_data["LA_acc_z_mean"])
	
    seg_data["LA_acc_z_var"]=np.var(LA_acc_z)
    Collect.append(seg_data["LA_acc_z_var"])
	
    seg_data["LA_acc_z_ptp"]=np.ptp(LA_acc_z)
    Collect.append(seg_data["LA_acc_z_ptp"])
    
    if index<=2398:
        seg_data["LA_acc_z_corrcoef"]=corrcoef(LA_acc_z,np.array(split_data[index+1]["LA_acc_z"]))
        Collect.append(seg_data["LA_acc_z_corrcoef"])
    else:
        seg_data["LA_acc_z_corrcoef"]=corrcoef(LA_acc_z,np.array(init_LA_acc_z))
        Collect.append(seg_data["LA_acc_z_corrcoef"])
    skew,kurt=calc_stat(LA_acc_z)
    seg_data["LA_acc_z_skew"]=skew
    Collect.append(seg_data["LA_acc_z_skew"])
    seg_data["LA_acc_z_kurt"]=kurt
    Collect.append(seg_data["LA_acc_z_kurt"])
    
    peak=np.argsort(-np.fft.fft(LA_acc_z))
    seg_data["LA_acc_z_peak1"]=peak[0]
    Collect.append(seg_data["LA_acc_z_peak1"])
    
    seg_data["LA_acc_z_peak2"]=peak[1]
    Collect.append(seg_data["LA_acc_z_peak2"])
	
    seg_data["LA_acc_z_peak3"]=peak[2]
    Collect.append(seg_data["LA_acc_z_peak3"])	

    seg_data["LA_acc_z_peak4"]=peak[3]
    Collect.append(seg_data["LA_acc_z_peak4"])
	
    seg_data["LA_acc_z_peak5"]=peak[4]
    Collect.append(seg_data["LA_acc_z_peak5"])
	
    autocor=autocorrelation(LA_acc_z,12)
    seg_data["LA_acc_z_autocor0"]=autocor[0]
    Collect.append(seg_data["LA_acc_z_autocor0"])
    
    seg_data["LA_acc_z_autocor1"]=autocor[1]
    Collect.append(seg_data["LA_acc_z_autocor1"])
	
    seg_data["LA_acc_z_autocor2"]=autocor[2]
    Collect.append(seg_data["LA_acc_z_autocor2"])
    
    seg_data["LA_acc_z_autocor3"]=autocor[3]
    Collect.append(seg_data["LA_acc_z_autocor3"])
	
    seg_data["LA_acc_z_autocor4"]=autocor[4]
    Collect.append(seg_data["LA_acc_z_autocor4"])
    
    seg_data["LA_acc_z_autocor5"]=autocor[5]
    Collect.append(seg_data["LA_acc_z_autocor5"])
	
    seg_data["LA_acc_z_autocor6"]=autocor[6]
    Collect.append(seg_data["LA_acc_z_autocor6"])
    
    seg_data["LA_acc_z_autocor7"]=autocor[7]
    Collect.append(seg_data["LA_acc_z_autocor7"])
	
    seg_data["LA_acc_z_autocor8"]=autocor[8]
    Collect.append(seg_data["LA_acc_z_autocor8"])
    
    seg_data["LA_acc_z_autocor9"]=autocor[9]
    Collect.append(seg_data["LA_acc_z_autocor9"])
	
    RL_acc_x=np.array(seg_data.pop("RL_acc_x")) 
    seg_data["RL_acc_x_mean"]=np.mean(RL_acc_x)
    Collect.append(seg_data["RL_acc_x_mean"])
	
    seg_data["RL_acc_x_var"]=np.var(RL_acc_x)
    Collect.append(seg_data["RL_acc_x_var"])
	
    seg_data["RL_acc_x_ptp"]=np.ptp(RL_acc_x)
    Collect.append(seg_data["RL_acc_x_ptp"])
    
    if index<=2398:
        seg_data["RL_acc_x_corrcoef"]=corrcoef(RL_acc_x,np.array(split_data[index+1]["RL_acc_x"]))
        Collect.append(seg_data["RL_acc_x_corrcoef"])
    else:
        seg_data["RL_acc_x_corrcoef"]=corrcoef(RL_acc_x,np.array(init_RL_acc_x))
        Collect.append(seg_data["RL_acc_x_corrcoef"])
    skew,kurt=calc_stat(RL_acc_x)
    seg_data["RL_acc_x_skew"]=skew
    Collect.append(seg_data["RL_acc_x_skew"])
    seg_data["RL_acc_x_kurt"]=kurt
    Collect.append(seg_data["RL_acc_x_kurt"])
    
    peak=np.argsort(-np.fft.fft(RL_acc_x))
    seg_data["RL_acc_x_peak1"]=peak[0]
    Collect.append(seg_data["RL_acc_x_peak1"])
    
    seg_data["RL_acc_x_peak2"]=peak[1]
    Collect.append(seg_data["RL_acc_x_peak2"])
	
    seg_data["RL_acc_x_peak3"]=peak[2]
    Collect.append(seg_data["RL_acc_x_peak3"])	

    seg_data["RL_acc_x_peak4"]=peak[3]
    Collect.append(seg_data["RL_acc_x_peak4"])
	
    seg_data["RL_acc_x_peak5"]=peak[4]
    Collect.append(seg_data["RL_acc_x_peak5"])
	
    autocor=autocorrelation(RL_acc_x,12)
    seg_data["RL_acc_x_autocor0"]=autocor[0]
    Collect.append(seg_data["RL_acc_x_autocor0"])
    
    seg_data["RL_acc_x_autocor1"]=autocor[1]
    Collect.append(seg_data["RL_acc_x_autocor1"])
	
    seg_data["RL_acc_x_autocor2"]=autocor[2]
    Collect.append(seg_data["RL_acc_x_autocor2"])
    
    seg_data["RL_acc_x_autocor3"]=autocor[3]
    Collect.append(seg_data["RL_acc_x_autocor3"])
	
    seg_data["RL_acc_x_autocor4"]=autocor[4]
    Collect.append(seg_data["RL_acc_x_autocor4"])
    
    seg_data["RL_acc_x_autocor5"]=autocor[5]
    Collect.append(seg_data["RL_acc_x_autocor5"])
	
    seg_data["RL_acc_x_autocor6"]=autocor[6]
    Collect.append(seg_data["RL_acc_x_autocor6"])
    
    seg_data["RL_acc_x_autocor7"]=autocor[7]
    Collect.append(seg_data["RL_acc_x_autocor7"])
	
    seg_data["RL_acc_x_autocor8"]=autocor[8]
    Collect.append(seg_data["RL_acc_x_autocor8"])
    
    seg_data["RL_acc_x_autocor9"]=autocor[9]
    Collect.append(seg_data["RL_acc_x_autocor9"])
	
    RL_acc_y=np.array(seg_data.pop("RL_acc_y")) 
    seg_data["RL_acc_y_mean"]=np.mean(RL_acc_y)
    Collect.append(seg_data["RL_acc_y_mean"])
	
    seg_data["RL_acc_y_var"]=np.var(RL_acc_y)
    Collect.append(seg_data["RL_acc_y_var"])
	
    seg_data["RL_acc_y_ptp"]=np.ptp(RL_acc_y)
    Collect.append(seg_data["RL_acc_y_ptp"])
    
    if index<=2398:
        seg_data["RL_acc_y_corrcoef"]=corrcoef(RL_acc_y,np.array(split_data[index+1]["RL_acc_y"]))
        Collect.append(seg_data["RL_acc_y_corrcoef"])
    else:
        seg_data["RL_acc_y_corrcoef"]=corrcoef(RL_acc_y,np.array(init_RL_acc_y))
        Collect.append(seg_data["RL_acc_y_corrcoef"])
    skew,kurt=calc_stat(RL_acc_y)
    seg_data["RL_acc_y_skew"]=skew
    Collect.append(seg_data["RL_acc_y_skew"])
    seg_data["RL_acc_y_kurt"]=kurt
    Collect.append(seg_data["RL_acc_y_kurt"])
    
    peak=np.argsort(-np.fft.fft(RL_acc_y))
    seg_data["RL_acc_y_peak1"]=peak[0]
    Collect.append(seg_data["RL_acc_y_peak1"])
    
    seg_data["RL_acc_y_peak2"]=peak[1]
    Collect.append(seg_data["RL_acc_y_peak2"])
	
    seg_data["RL_acc_y_peak3"]=peak[2]
    Collect.append(seg_data["RL_acc_y_peak3"])	

    seg_data["RL_acc_y_peak4"]=peak[3]
    Collect.append(seg_data["RL_acc_y_peak4"])
	
    seg_data["RL_acc_y_peak5"]=peak[4]
    Collect.append(seg_data["RL_acc_y_peak5"])
	
    autocor=autocorrelation(RL_acc_y,12)
    seg_data["RL_acc_y_autocor0"]=autocor[0]
    Collect.append(seg_data["RL_acc_y_autocor0"])
    
    seg_data["RL_acc_y_autocor1"]=autocor[1]
    Collect.append(seg_data["RL_acc_y_autocor1"])
	
    seg_data["RL_acc_y_autocor2"]=autocor[2]
    Collect.append(seg_data["RL_acc_y_autocor2"])
    
    seg_data["RL_acc_y_autocor3"]=autocor[3]
    Collect.append(seg_data["RL_acc_y_autocor3"])
	
    seg_data["RL_acc_y_autocor4"]=autocor[4]
    Collect.append(seg_data["RL_acc_y_autocor4"])
    
    seg_data["RL_acc_y_autocor5"]=autocor[5]
    Collect.append(seg_data["RL_acc_y_autocor5"])
	
    seg_data["RL_acc_y_autocor6"]=autocor[6]
    Collect.append(seg_data["RL_acc_y_autocor6"])
    
    seg_data["RL_acc_y_autocor7"]=autocor[7]
    Collect.append(seg_data["RL_acc_y_autocor7"])
	
    seg_data["RL_acc_y_autocor8"]=autocor[8]
    Collect.append(seg_data["RL_acc_y_autocor8"])
    
    seg_data["RL_acc_y_autocor9"]=autocor[9]
    Collect.append(seg_data["RL_acc_y_autocor9"])
	
    RL_acc_z=np.array(seg_data.pop("RL_acc_z")) 
    seg_data["RL_acc_z_mean"]=np.mean(RL_acc_z)
    Collect.append(seg_data["RL_acc_z_mean"])
	
    seg_data["RL_acc_z_var"]=np.var(RL_acc_z)
    Collect.append(seg_data["RL_acc_z_var"])
	
    seg_data["RL_acc_z_ptp"]=np.ptp(RL_acc_z)
    Collect.append(seg_data["RL_acc_z_ptp"])
    
    if index<=2398:
        seg_data["RL_acc_z_corrcoef"]=corrcoef(RL_acc_z,np.array(split_data[index+1]["RL_acc_z"]))
        Collect.append(seg_data["RL_acc_z_corrcoef"])
    else:
        seg_data["RL_acc_z_corrcoef"]=corrcoef(RL_acc_z,np.array(init_RL_acc_z))
        Collect.append(seg_data["RL_acc_z_corrcoef"])
    skew,kurt=calc_stat(RL_acc_z)
    seg_data["RL_acc_z_skew"]=skew
    Collect.append(seg_data["RL_acc_z_skew"])
    seg_data["RL_acc_z_kurt"]=kurt
    Collect.append(seg_data["RL_acc_z_kurt"])
    
    peak=np.argsort(-np.fft.fft(RL_acc_z))
    seg_data["RL_acc_z_peak1"]=peak[0]
    Collect.append(seg_data["RL_acc_z_peak1"])
    
    seg_data["RL_acc_z_peak2"]=peak[1]
    Collect.append(seg_data["RL_acc_z_peak2"])
	
    seg_data["RL_acc_z_peak3"]=peak[2]
    Collect.append(seg_data["RL_acc_z_peak3"])	

    seg_data["RL_acc_z_peak4"]=peak[3]
    Collect.append(seg_data["RL_acc_z_peak4"])
	
    seg_data["RL_acc_z_peak5"]=peak[4]
    Collect.append(seg_data["RL_acc_z_peak5"])
	
    autocor=autocorrelation(RL_acc_z,12)
    seg_data["RL_acc_z_autocor0"]=autocor[0]
    Collect.append(seg_data["RL_acc_z_autocor0"])
    
    seg_data["RL_acc_z_autocor1"]=autocor[1]
    Collect.append(seg_data["RL_acc_z_autocor1"])
	
    seg_data["RL_acc_z_autocor2"]=autocor[2]
    Collect.append(seg_data["RL_acc_z_autocor2"])
    
    seg_data["RL_acc_z_autocor3"]=autocor[3]
    Collect.append(seg_data["RL_acc_z_autocor3"])
	
    seg_data["RL_acc_z_autocor4"]=autocor[4]
    Collect.append(seg_data["RL_acc_z_autocor4"])
    
    seg_data["RL_acc_z_autocor5"]=autocor[5]
    Collect.append(seg_data["RL_acc_z_autocor5"])
	
    seg_data["RL_acc_z_autocor6"]=autocor[6]
    Collect.append(seg_data["RL_acc_z_autocor6"])
    
    seg_data["RL_acc_z_autocor7"]=autocor[7]
    Collect.append(seg_data["RL_acc_z_autocor7"])
	
    seg_data["RL_acc_z_autocor8"]=autocor[8]
    Collect.append(seg_data["RL_acc_z_autocor8"])
    
    seg_data["RL_acc_z_autocor9"]=autocor[9]
    Collect.append(seg_data["RL_acc_z_autocor9"])
	
    LL_acc_x=np.array(seg_data.pop("LL_acc_x")) 
    seg_data["LL_acc_x_mean"]=np.mean(LL_acc_x)
    Collect.append(seg_data["LL_acc_x_mean"])
	
    seg_data["LL_acc_x_var"]=np.var(LL_acc_x)
    Collect.append(seg_data["LL_acc_x_var"])
	
    seg_data["LL_acc_x_ptp"]=np.ptp(LL_acc_x)
    Collect.append(seg_data["LL_acc_x_ptp"])
    
    if index<=2398:
        seg_data["LL_acc_x_corrcoef"]=corrcoef(LL_acc_x,np.array(split_data[index+1]["LL_acc_x"]))
        Collect.append(seg_data["LL_acc_x_corrcoef"])
    else:
        seg_data["LL_acc_x_corrcoef"]=corrcoef(LL_acc_x,np.array(init_LL_acc_x))
        Collect.append(seg_data["LL_acc_x_corrcoef"])
    skew,kurt=calc_stat(LL_acc_x)
    seg_data["LL_acc_x_skew"]=skew
    Collect.append(seg_data["LL_acc_x_skew"])
    seg_data["LL_acc_x_kurt"]=kurt
    Collect.append(seg_data["LL_acc_x_kurt"])
    
    peak=np.argsort(-np.fft.fft(LL_acc_x))
    seg_data["LL_acc_x_peak1"]=peak[0]
    Collect.append(seg_data["LL_acc_x_peak1"])
    
    seg_data["LL_acc_x_peak2"]=peak[1]
    Collect.append(seg_data["LL_acc_x_peak2"])
	
    seg_data["LL_acc_x_peak3"]=peak[2]
    Collect.append(seg_data["LL_acc_x_peak3"])	

    seg_data["LL_acc_x_peak4"]=peak[3]
    Collect.append(seg_data["LL_acc_x_peak4"])
	
    seg_data["LL_acc_x_peak5"]=peak[4]
    Collect.append(seg_data["LL_acc_x_peak5"])
	
    autocor=autocorrelation(LL_acc_x,12)
    seg_data["LL_acc_x_autocor0"]=autocor[0]
    Collect.append(seg_data["LL_acc_x_autocor0"])
    
    seg_data["LL_acc_x_autocor1"]=autocor[1]
    Collect.append(seg_data["LL_acc_x_autocor1"])
	
    seg_data["LL_acc_x_autocor2"]=autocor[2]
    Collect.append(seg_data["LL_acc_x_autocor2"])
    
    seg_data["LL_acc_x_autocor3"]=autocor[3]
    Collect.append(seg_data["LL_acc_x_autocor3"])
	
    seg_data["LL_acc_x_autocor4"]=autocor[4]
    Collect.append(seg_data["LL_acc_x_autocor4"])
    
    seg_data["LL_acc_x_autocor5"]=autocor[5]
    Collect.append(seg_data["LL_acc_x_autocor5"])
	
    seg_data["LL_acc_x_autocor6"]=autocor[6]
    Collect.append(seg_data["LL_acc_x_autocor6"])
    
    seg_data["LL_acc_x_autocor7"]=autocor[7]
    Collect.append(seg_data["LL_acc_x_autocor7"])
	
    seg_data["LL_acc_x_autocor8"]=autocor[8]
    Collect.append(seg_data["LL_acc_x_autocor8"])
    
    seg_data["LL_acc_x_autocor9"]=autocor[9]
    Collect.append(seg_data["LL_acc_x_autocor9"])
	
    LL_acc_y=np.array(seg_data.pop("LL_acc_y")) 
    seg_data["LL_acc_y_mean"]=np.mean(LL_acc_y)
    Collect.append(seg_data["LL_acc_y_mean"])
	
    seg_data["LL_acc_y_var"]=np.var(LL_acc_y)
    Collect.append(seg_data["LL_acc_y_var"])
	
    seg_data["LL_acc_y_ptp"]=np.ptp(LL_acc_y)
    Collect.append(seg_data["LL_acc_y_ptp"])
    
    if index<=2398:
        seg_data["LL_acc_y_corrcoef"]=corrcoef(LL_acc_y,np.array(split_data[index+1]["LL_acc_y"]))
        Collect.append(seg_data["LL_acc_y_corrcoef"])
    else:
        seg_data["LL_acc_y_corrcoef"]=corrcoef(LL_acc_y,np.array(init_LL_acc_y))
        Collect.append(seg_data["LL_acc_y_corrcoef"])
    skew,kurt=calc_stat(LL_acc_y)
    seg_data["LL_acc_y_skew"]=skew
    Collect.append(seg_data["LL_acc_y_skew"])
    seg_data["LL_acc_y_kurt"]=kurt
    Collect.append(seg_data["LL_acc_y_kurt"])
    
    peak=np.argsort(-np.fft.fft(LL_acc_y))
    seg_data["LL_acc_y_peak1"]=peak[0]
    Collect.append(seg_data["LL_acc_y_peak1"])
    
    seg_data["LL_acc_y_peak2"]=peak[1]
    Collect.append(seg_data["LL_acc_y_peak2"])
	
    seg_data["LL_acc_y_peak3"]=peak[2]
    Collect.append(seg_data["LL_acc_y_peak3"])	

    seg_data["LL_acc_y_peak4"]=peak[3]
    Collect.append(seg_data["LL_acc_y_peak4"])
	
    seg_data["LL_acc_y_peak5"]=peak[4]
    Collect.append(seg_data["LL_acc_y_peak5"])
	
    autocor=autocorrelation(LL_acc_y,12)
    seg_data["LL_acc_y_autocor0"]=autocor[0]
    Collect.append(seg_data["LL_acc_y_autocor0"])
    
    seg_data["LL_acc_y_autocor1"]=autocor[1]
    Collect.append(seg_data["LL_acc_y_autocor1"])
	
    seg_data["LL_acc_y_autocor2"]=autocor[2]
    Collect.append(seg_data["LL_acc_y_autocor2"])
    
    seg_data["LL_acc_y_autocor3"]=autocor[3]
    Collect.append(seg_data["LL_acc_y_autocor3"])
	
    seg_data["LL_acc_y_autocor4"]=autocor[4]
    Collect.append(seg_data["LL_acc_y_autocor4"])
    
    seg_data["LL_acc_y_autocor5"]=autocor[5]
    Collect.append(seg_data["LL_acc_y_autocor5"])
	
    seg_data["LL_acc_y_autocor6"]=autocor[6]
    Collect.append(seg_data["LL_acc_y_autocor6"])
    
    seg_data["LL_acc_y_autocor7"]=autocor[7]
    Collect.append(seg_data["LL_acc_y_autocor7"])
	
    seg_data["LL_acc_y_autocor8"]=autocor[8]
    Collect.append(seg_data["LL_acc_y_autocor8"])
    
    seg_data["LL_acc_y_autocor9"]=autocor[9]
    Collect.append(seg_data["LL_acc_y_autocor9"])
	
    LL_acc_z=np.array(seg_data.pop("LL_acc_z")) 
    seg_data["LL_acc_z_mean"]=np.mean(LL_acc_z)
    Collect.append(seg_data["LL_acc_z_mean"])
	
    seg_data["LL_acc_z_var"]=np.var(LL_acc_z)
    Collect.append(seg_data["LL_acc_z_var"])
	
    seg_data["LL_acc_z_ptp"]=np.ptp(LL_acc_z)
    Collect.append(seg_data["LL_acc_z_ptp"])
    
    if index<=2398:
        seg_data["LL_acc_z_corrcoef"]=corrcoef(LL_acc_z,np.array(split_data[index+1]["LL_acc_z"]))
        Collect.append(seg_data["LL_acc_z_corrcoef"])
    else:
        seg_data["LL_acc_z_corrcoef"]=corrcoef(LL_acc_z,np.array(init_LL_acc_z))
        Collect.append(seg_data["LL_acc_z_corrcoef"])
    skew,kurt=calc_stat(LL_acc_z)
    seg_data["LL_acc_z_skew"]=skew
    Collect.append(seg_data["LL_acc_z_skew"])
    seg_data["LL_acc_z_kurt"]=kurt
    Collect.append(seg_data["LL_acc_z_kurt"])
    
    peak=np.argsort(-np.fft.fft(LL_acc_z))
    seg_data["LL_acc_z_peak1"]=peak[0]
    Collect.append(seg_data["LL_acc_z_peak1"])
    
    seg_data["LL_acc_z_peak2"]=peak[1]
    Collect.append(seg_data["LL_acc_z_peak2"])
	
    seg_data["LL_acc_z_peak3"]=peak[2]
    Collect.append(seg_data["LL_acc_z_peak3"])	

    seg_data["LL_acc_z_peak4"]=peak[3]
    Collect.append(seg_data["LL_acc_z_peak4"])
	
    seg_data["LL_acc_z_peak5"]=peak[4]
    Collect.append(seg_data["LL_acc_z_peak5"])
	
    autocor=autocorrelation(LL_acc_z,12)
    seg_data["LL_acc_z_autocor0"]=autocor[0]
    Collect.append(seg_data["LL_acc_z_autocor0"])
    
    seg_data["LL_acc_z_autocor1"]=autocor[1]
    Collect.append(seg_data["LL_acc_z_autocor1"])
	
    seg_data["LL_acc_z_autocor2"]=autocor[2]
    Collect.append(seg_data["LL_acc_z_autocor2"])
    
    seg_data["LL_acc_z_autocor3"]=autocor[3]
    Collect.append(seg_data["LL_acc_z_autocor3"])
	
    seg_data["LL_acc_z_autocor4"]=autocor[4]
    Collect.append(seg_data["LL_acc_z_autocor4"])
    
    seg_data["LL_acc_z_autocor5"]=autocor[5]
    Collect.append(seg_data["LL_acc_z_autocor5"])
	
    seg_data["LL_acc_z_autocor6"]=autocor[6]
    Collect.append(seg_data["LL_acc_z_autocor6"])
    
    seg_data["LL_acc_z_autocor7"]=autocor[7]
    Collect.append(seg_data["LL_acc_z_autocor7"])
	
    seg_data["LL_acc_z_autocor8"]=autocor[8]
    Collect.append(seg_data["LL_acc_z_autocor8"])
    
    seg_data["LL_acc_z_autocor9"]=autocor[9]
    Collect.append(seg_data["LL_acc_z_autocor9"])
    labels.append(seg_data["label"])
	
    statistics_data.append(seg_data)
#labels=np.array(labels)
#Collect_arr=np.array(Collect).reshape((2400,315))
#Collect_norm=preprocessing.normalize(Collect_arr,norm="l2") #标准化
random.shuffle(statistics_data) #随机打乱顺序
random.shuffle(statistics_data)
Collect_rd=[]
labels_rd=[]
for seg_data in statistics_data:      #分割多个子集
    Collect_rd.append(seg_data["T_acc_x_mean"])
    Collect_rd.append(seg_data["T_acc_x_var"])
    Collect_rd.append(seg_data["T_acc_x_ptp"])
    Collect_rd.append(seg_data["T_acc_x_corrcoef"])
    Collect_rd.append(seg_data["T_acc_x_skew"])
    Collect_rd.append(seg_data["T_acc_x_kurt"])
    Collect_rd.append(seg_data["T_acc_x_peak1"])
    Collect_rd.append(seg_data["T_acc_x_peak2"])
    Collect_rd.append(seg_data["T_acc_x_peak3"])	
    Collect_rd.append(seg_data["T_acc_x_peak4"])
    Collect_rd.append(seg_data["T_acc_x_peak5"])
    Collect_rd.append(seg_data["T_acc_x_autocor0"])   
    Collect_rd.append(seg_data["T_acc_x_autocor1"])
    Collect_rd.append(seg_data["T_acc_x_autocor2"])
    Collect_rd.append(seg_data["T_acc_x_autocor3"])
    Collect_rd.append(seg_data["T_acc_x_autocor4"])        
    Collect_rd.append(seg_data["T_acc_x_autocor5"])
    Collect_rd.append(seg_data["T_acc_x_autocor6"])   
    Collect_rd.append(seg_data["T_acc_x_autocor7"])
    Collect_rd.append(seg_data["T_acc_x_autocor8"])    
    Collect_rd.append(seg_data["T_acc_x_autocor9"])	
    
    Collect_rd.append(seg_data["T_acc_y_mean"])    
    Collect_rd.append(seg_data["T_acc_y_var"])    
    Collect_rd.append(seg_data["T_acc_y_ptp"])
    Collect_rd.append(seg_data["T_acc_y_corrcoef"])    
    Collect_rd.append(seg_data["T_acc_y_skew"])    
    Collect_rd.append(seg_data["T_acc_y_kurt"])    
    Collect_rd.append(seg_data["T_acc_y_peak1"])    
    Collect_rd.append(seg_data["T_acc_y_peak2"])    
    Collect_rd.append(seg_data["T_acc_y_peak3"])	    
    Collect_rd.append(seg_data["T_acc_y_peak4"])   
    Collect_rd.append(seg_data["T_acc_y_peak5"])    
    Collect_rd.append(seg_data["T_acc_y_autocor0"])    
    Collect_rd.append(seg_data["T_acc_y_autocor1"])    
    Collect_rd.append(seg_data["T_acc_y_autocor2"])    
    Collect_rd.append(seg_data["T_acc_y_autocor3"])    
    Collect_rd.append(seg_data["T_acc_y_autocor4"])    
    Collect_rd.append(seg_data["T_acc_y_autocor5"])    
    Collect_rd.append(seg_data["T_acc_y_autocor6"])    
    Collect_rd.append(seg_data["T_acc_y_autocor7"])    
    Collect_rd.append(seg_data["T_acc_y_autocor8"])   
    Collect_rd.append(seg_data["T_acc_y_autocor9"]) 
	
    Collect_rd.append(seg_data["T_acc_z_mean"])    
    Collect_rd.append(seg_data["T_acc_z_var"])    
    Collect_rd.append(seg_data["T_acc_z_ptp"])  
    Collect_rd.append(seg_data["T_acc_z_corrcoef"])   
    Collect_rd.append(seg_data["T_acc_z_skew"])    
    Collect_rd.append(seg_data["T_acc_z_kurt"])   
    Collect_rd.append(seg_data["T_acc_z_peak1"])    
    Collect_rd.append(seg_data["T_acc_z_peak2"])    
    Collect_rd.append(seg_data["T_acc_z_peak3"])	   
    Collect_rd.append(seg_data["T_acc_z_peak4"])   
    Collect_rd.append(seg_data["T_acc_z_peak5"])    
    Collect_rd.append(seg_data["T_acc_z_autocor0"])    
    Collect_rd.append(seg_data["T_acc_z_autocor1"])    
    Collect_rd.append(seg_data["T_acc_z_autocor2"])    
    Collect_rd.append(seg_data["T_acc_z_autocor3"])
    Collect_rd.append(seg_data["T_acc_z_autocor4"])    
    Collect_rd.append(seg_data["T_acc_z_autocor5"])    
    Collect_rd.append(seg_data["T_acc_z_autocor6"])    
    Collect_rd.append(seg_data["T_acc_z_autocor7"])    
    Collect_rd.append(seg_data["T_acc_z_autocor8"])
    Collect_rd.append(seg_data["T_acc_z_autocor9"])
	
    Collect_rd.append(seg_data["RA_acc_x_mean"])     ####################
    Collect_rd.append(seg_data["RA_acc_x_var"])
    Collect_rd.append(seg_data["RA_acc_x_ptp"])
    Collect_rd.append(seg_data["RA_acc_x_corrcoef"])
    Collect_rd.append(seg_data["RA_acc_x_skew"])
    Collect_rd.append(seg_data["RA_acc_x_kurt"])
    Collect_rd.append(seg_data["RA_acc_x_peak1"])
    Collect_rd.append(seg_data["RA_acc_x_peak2"])
    Collect_rd.append(seg_data["RA_acc_x_peak3"])	
    Collect_rd.append(seg_data["RA_acc_x_peak4"])
    Collect_rd.append(seg_data["RA_acc_x_peak5"])
    Collect_rd.append(seg_data["RA_acc_x_autocor0"])   
    Collect_rd.append(seg_data["RA_acc_x_autocor1"])
    Collect_rd.append(seg_data["RA_acc_x_autocor2"])
    Collect_rd.append(seg_data["RA_acc_x_autocor3"])
    Collect_rd.append(seg_data["RA_acc_x_autocor4"])        
    Collect_rd.append(seg_data["RA_acc_x_autocor5"])
    Collect_rd.append(seg_data["RA_acc_x_autocor6"])   
    Collect_rd.append(seg_data["RA_acc_x_autocor7"])
    Collect_rd.append(seg_data["RA_acc_x_autocor8"])    
    Collect_rd.append(seg_data["RA_acc_x_autocor9"])	
    
    Collect_rd.append(seg_data["RA_acc_y_mean"])    
    Collect_rd.append(seg_data["RA_acc_y_var"])    
    Collect_rd.append(seg_data["RA_acc_y_ptp"])
    Collect_rd.append(seg_data["RA_acc_y_corrcoef"])    
    Collect_rd.append(seg_data["RA_acc_y_skew"])    
    Collect_rd.append(seg_data["RA_acc_y_kurt"])    
    Collect_rd.append(seg_data["RA_acc_y_peak1"])    
    Collect_rd.append(seg_data["RA_acc_y_peak2"])    
    Collect_rd.append(seg_data["RA_acc_y_peak3"])	    
    Collect_rd.append(seg_data["RA_acc_y_peak4"])   
    Collect_rd.append(seg_data["RA_acc_y_peak5"])    
    Collect_rd.append(seg_data["RA_acc_y_autocor0"])    
    Collect_rd.append(seg_data["RA_acc_y_autocor1"])    
    Collect_rd.append(seg_data["RA_acc_y_autocor2"])    
    Collect_rd.append(seg_data["RA_acc_y_autocor3"])    
    Collect_rd.append(seg_data["RA_acc_y_autocor4"])    
    Collect_rd.append(seg_data["RA_acc_y_autocor5"])    
    Collect_rd.append(seg_data["RA_acc_y_autocor6"])    
    Collect_rd.append(seg_data["RA_acc_y_autocor7"])    
    Collect_rd.append(seg_data["RA_acc_y_autocor8"])   
    Collect_rd.append(seg_data["RA_acc_y_autocor9"]) 
	
    Collect_rd.append(seg_data["RA_acc_z_mean"])    
    Collect_rd.append(seg_data["RA_acc_z_var"])    
    Collect_rd.append(seg_data["RA_acc_z_ptp"])  
    Collect_rd.append(seg_data["RA_acc_z_corrcoef"])   
    Collect_rd.append(seg_data["RA_acc_z_skew"])    
    Collect_rd.append(seg_data["RA_acc_z_kurt"])   
    Collect_rd.append(seg_data["RA_acc_z_peak1"])    
    Collect_rd.append(seg_data["RA_acc_z_peak2"])    
    Collect_rd.append(seg_data["RA_acc_z_peak3"])	   
    Collect_rd.append(seg_data["RA_acc_z_peak4"])   
    Collect_rd.append(seg_data["RA_acc_z_peak5"])    
    Collect_rd.append(seg_data["RA_acc_z_autocor0"])    
    Collect_rd.append(seg_data["RA_acc_z_autocor1"])    
    Collect_rd.append(seg_data["RA_acc_z_autocor2"])    
    Collect_rd.append(seg_data["RA_acc_z_autocor3"])
    Collect_rd.append(seg_data["RA_acc_z_autocor4"])    
    Collect_rd.append(seg_data["RA_acc_z_autocor5"])    
    Collect_rd.append(seg_data["RA_acc_z_autocor6"])    
    Collect_rd.append(seg_data["RA_acc_z_autocor7"])    
    Collect_rd.append(seg_data["RA_acc_z_autocor8"])
    Collect_rd.append(seg_data["RA_acc_z_autocor9"]) 

    Collect_rd.append(seg_data["LA_acc_x_mean"])     ####################
    Collect_rd.append(seg_data["LA_acc_x_var"])
    Collect_rd.append(seg_data["LA_acc_x_ptp"])
    Collect_rd.append(seg_data["LA_acc_x_corrcoef"])
    Collect_rd.append(seg_data["LA_acc_x_skew"])
    Collect_rd.append(seg_data["LA_acc_x_kurt"])
    Collect_rd.append(seg_data["LA_acc_x_peak1"])
    Collect_rd.append(seg_data["LA_acc_x_peak2"])
    Collect_rd.append(seg_data["LA_acc_x_peak3"])	
    Collect_rd.append(seg_data["LA_acc_x_peak4"])
    Collect_rd.append(seg_data["LA_acc_x_peak5"])
    Collect_rd.append(seg_data["LA_acc_x_autocor0"])   
    Collect_rd.append(seg_data["LA_acc_x_autocor1"])
    Collect_rd.append(seg_data["LA_acc_x_autocor2"])
    Collect_rd.append(seg_data["LA_acc_x_autocor3"])
    Collect_rd.append(seg_data["LA_acc_x_autocor4"])        
    Collect_rd.append(seg_data["LA_acc_x_autocor5"])
    Collect_rd.append(seg_data["LA_acc_x_autocor6"])   
    Collect_rd.append(seg_data["LA_acc_x_autocor7"])
    Collect_rd.append(seg_data["LA_acc_x_autocor8"])    
    Collect_rd.append(seg_data["LA_acc_x_autocor9"])	
    
    Collect_rd.append(seg_data["LA_acc_y_mean"])    
    Collect_rd.append(seg_data["LA_acc_y_var"])    
    Collect_rd.append(seg_data["LA_acc_y_ptp"])
    Collect_rd.append(seg_data["LA_acc_y_corrcoef"])    
    Collect_rd.append(seg_data["LA_acc_y_skew"])    
    Collect_rd.append(seg_data["LA_acc_y_kurt"])    
    Collect_rd.append(seg_data["LA_acc_y_peak1"])    
    Collect_rd.append(seg_data["LA_acc_y_peak2"])    
    Collect_rd.append(seg_data["LA_acc_y_peak3"])	    
    Collect_rd.append(seg_data["LA_acc_y_peak4"])   
    Collect_rd.append(seg_data["LA_acc_y_peak5"])    
    Collect_rd.append(seg_data["LA_acc_y_autocor0"])    
    Collect_rd.append(seg_data["LA_acc_y_autocor1"])    
    Collect_rd.append(seg_data["LA_acc_y_autocor2"])    
    Collect_rd.append(seg_data["LA_acc_y_autocor3"])    
    Collect_rd.append(seg_data["LA_acc_y_autocor4"])    
    Collect_rd.append(seg_data["LA_acc_y_autocor5"])    
    Collect_rd.append(seg_data["LA_acc_y_autocor6"])    
    Collect_rd.append(seg_data["LA_acc_y_autocor7"])    
    Collect_rd.append(seg_data["LA_acc_y_autocor8"])   
    Collect_rd.append(seg_data["LA_acc_y_autocor9"]) 
	
    Collect_rd.append(seg_data["LA_acc_z_mean"])    
    Collect_rd.append(seg_data["LA_acc_z_var"])    
    Collect_rd.append(seg_data["LA_acc_z_ptp"])  
    Collect_rd.append(seg_data["LA_acc_z_corrcoef"])   
    Collect_rd.append(seg_data["LA_acc_z_skew"])    
    Collect_rd.append(seg_data["LA_acc_z_kurt"])   
    Collect_rd.append(seg_data["LA_acc_z_peak1"])    
    Collect_rd.append(seg_data["LA_acc_z_peak2"])    
    Collect_rd.append(seg_data["LA_acc_z_peak3"])	   
    Collect_rd.append(seg_data["LA_acc_z_peak4"])   
    Collect_rd.append(seg_data["LA_acc_z_peak5"])    
    Collect_rd.append(seg_data["LA_acc_z_autocor0"])    
    Collect_rd.append(seg_data["LA_acc_z_autocor1"])    
    Collect_rd.append(seg_data["LA_acc_z_autocor2"])    
    Collect_rd.append(seg_data["LA_acc_z_autocor3"])
    Collect_rd.append(seg_data["LA_acc_z_autocor4"])    
    Collect_rd.append(seg_data["LA_acc_z_autocor5"])    
    Collect_rd.append(seg_data["LA_acc_z_autocor6"])    
    Collect_rd.append(seg_data["LA_acc_z_autocor7"])    
    Collect_rd.append(seg_data["LA_acc_z_autocor8"])
    Collect_rd.append(seg_data["LA_acc_z_autocor9"]) 	
	
    Collect_rd.append(seg_data["RL_acc_x_mean"])     ####################
    Collect_rd.append(seg_data["RL_acc_x_var"])
    Collect_rd.append(seg_data["RL_acc_x_ptp"])
    Collect_rd.append(seg_data["RL_acc_x_corrcoef"])
    Collect_rd.append(seg_data["RL_acc_x_skew"])
    Collect_rd.append(seg_data["RL_acc_x_kurt"])
    Collect_rd.append(seg_data["RL_acc_x_peak1"])
    Collect_rd.append(seg_data["RL_acc_x_peak2"])
    Collect_rd.append(seg_data["RL_acc_x_peak3"])	
    Collect_rd.append(seg_data["RL_acc_x_peak4"])
    Collect_rd.append(seg_data["RL_acc_x_peak5"])
    Collect_rd.append(seg_data["RL_acc_x_autocor0"])   
    Collect_rd.append(seg_data["RL_acc_x_autocor1"])
    Collect_rd.append(seg_data["RL_acc_x_autocor2"])
    Collect_rd.append(seg_data["RL_acc_x_autocor3"])
    Collect_rd.append(seg_data["RL_acc_x_autocor4"])        
    Collect_rd.append(seg_data["RL_acc_x_autocor5"])
    Collect_rd.append(seg_data["RL_acc_x_autocor6"])   
    Collect_rd.append(seg_data["RL_acc_x_autocor7"])
    Collect_rd.append(seg_data["RL_acc_x_autocor8"])    
    Collect_rd.append(seg_data["RL_acc_x_autocor9"])	
    
    Collect_rd.append(seg_data["RL_acc_y_mean"])    
    Collect_rd.append(seg_data["RL_acc_y_var"])    
    Collect_rd.append(seg_data["RL_acc_y_ptp"])
    Collect_rd.append(seg_data["RL_acc_y_corrcoef"])    
    Collect_rd.append(seg_data["RL_acc_y_skew"])    
    Collect_rd.append(seg_data["RL_acc_y_kurt"])    
    Collect_rd.append(seg_data["RL_acc_y_peak1"])    
    Collect_rd.append(seg_data["RL_acc_y_peak2"])    
    Collect_rd.append(seg_data["RL_acc_y_peak3"])	    
    Collect_rd.append(seg_data["RL_acc_y_peak4"])   
    Collect_rd.append(seg_data["RL_acc_y_peak5"])    
    Collect_rd.append(seg_data["RL_acc_y_autocor0"])    
    Collect_rd.append(seg_data["RL_acc_y_autocor1"])    
    Collect_rd.append(seg_data["RL_acc_y_autocor2"])    
    Collect_rd.append(seg_data["RL_acc_y_autocor3"])    
    Collect_rd.append(seg_data["RL_acc_y_autocor4"])    
    Collect_rd.append(seg_data["RL_acc_y_autocor5"])    
    Collect_rd.append(seg_data["RL_acc_y_autocor6"])    
    Collect_rd.append(seg_data["RL_acc_y_autocor7"])    
    Collect_rd.append(seg_data["RL_acc_y_autocor8"])   
    Collect_rd.append(seg_data["RL_acc_y_autocor9"]) 
	
    Collect_rd.append(seg_data["RL_acc_z_mean"])    
    Collect_rd.append(seg_data["RL_acc_z_var"])    
    Collect_rd.append(seg_data["RL_acc_z_ptp"])  
    Collect_rd.append(seg_data["RL_acc_z_corrcoef"])   
    Collect_rd.append(seg_data["RL_acc_z_skew"])    
    Collect_rd.append(seg_data["RL_acc_z_kurt"])   
    Collect_rd.append(seg_data["RL_acc_z_peak1"])    
    Collect_rd.append(seg_data["RL_acc_z_peak2"])    
    Collect_rd.append(seg_data["RL_acc_z_peak3"])	   
    Collect_rd.append(seg_data["RL_acc_z_peak4"])   
    Collect_rd.append(seg_data["RL_acc_z_peak5"])    
    Collect_rd.append(seg_data["RL_acc_z_autocor0"])    
    Collect_rd.append(seg_data["RL_acc_z_autocor1"])    
    Collect_rd.append(seg_data["RL_acc_z_autocor2"])    
    Collect_rd.append(seg_data["RL_acc_z_autocor3"])
    Collect_rd.append(seg_data["RL_acc_z_autocor4"])    
    Collect_rd.append(seg_data["RL_acc_z_autocor5"])    
    Collect_rd.append(seg_data["RL_acc_z_autocor6"])    
    Collect_rd.append(seg_data["RL_acc_z_autocor7"])    
    Collect_rd.append(seg_data["RL_acc_z_autocor8"])
    Collect_rd.append(seg_data["RL_acc_z_autocor9"]) 
	
    Collect_rd.append(seg_data["LL_acc_x_mean"])     ####################
    Collect_rd.append(seg_data["LL_acc_x_var"])
    Collect_rd.append(seg_data["LL_acc_x_ptp"])
    Collect_rd.append(seg_data["LL_acc_x_corrcoef"])
    Collect_rd.append(seg_data["LL_acc_x_skew"])
    Collect_rd.append(seg_data["LL_acc_x_kurt"])
    Collect_rd.append(seg_data["LL_acc_x_peak1"])
    Collect_rd.append(seg_data["LL_acc_x_peak2"])
    Collect_rd.append(seg_data["LL_acc_x_peak3"])	
    Collect_rd.append(seg_data["LL_acc_x_peak4"])
    Collect_rd.append(seg_data["LL_acc_x_peak5"])
    Collect_rd.append(seg_data["LL_acc_x_autocor0"])   
    Collect_rd.append(seg_data["LL_acc_x_autocor1"])
    Collect_rd.append(seg_data["LL_acc_x_autocor2"])
    Collect_rd.append(seg_data["LL_acc_x_autocor3"])
    Collect_rd.append(seg_data["LL_acc_x_autocor4"])        
    Collect_rd.append(seg_data["LL_acc_x_autocor5"])
    Collect_rd.append(seg_data["LL_acc_x_autocor6"])   
    Collect_rd.append(seg_data["LL_acc_x_autocor7"])
    Collect_rd.append(seg_data["LL_acc_x_autocor8"])    
    Collect_rd.append(seg_data["LL_acc_x_autocor9"])	
    
    Collect_rd.append(seg_data["LL_acc_y_mean"])    
    Collect_rd.append(seg_data["LL_acc_y_var"])    
    Collect_rd.append(seg_data["LL_acc_y_ptp"])
    Collect_rd.append(seg_data["LL_acc_y_corrcoef"])    
    Collect_rd.append(seg_data["LL_acc_y_skew"])    
    Collect_rd.append(seg_data["LL_acc_y_kurt"])    
    Collect_rd.append(seg_data["LL_acc_y_peak1"])    
    Collect_rd.append(seg_data["LL_acc_y_peak2"])    
    Collect_rd.append(seg_data["LL_acc_y_peak3"])	    
    Collect_rd.append(seg_data["LL_acc_y_peak4"])   
    Collect_rd.append(seg_data["LL_acc_y_peak5"])    
    Collect_rd.append(seg_data["LL_acc_y_autocor0"])    
    Collect_rd.append(seg_data["LL_acc_y_autocor1"])    
    Collect_rd.append(seg_data["LL_acc_y_autocor2"])    
    Collect_rd.append(seg_data["LL_acc_y_autocor3"])    
    Collect_rd.append(seg_data["LL_acc_y_autocor4"])    
    Collect_rd.append(seg_data["LL_acc_y_autocor5"])    
    Collect_rd.append(seg_data["LL_acc_y_autocor6"])    
    Collect_rd.append(seg_data["LL_acc_y_autocor7"])    
    Collect_rd.append(seg_data["LL_acc_y_autocor8"])   
    Collect_rd.append(seg_data["LL_acc_y_autocor9"]) 
	
    Collect_rd.append(seg_data["LL_acc_z_mean"])    
    Collect_rd.append(seg_data["LL_acc_z_var"])    
    Collect_rd.append(seg_data["LL_acc_z_ptp"])  
    Collect_rd.append(seg_data["LL_acc_z_corrcoef"])   
    Collect_rd.append(seg_data["LL_acc_z_skew"])    
    Collect_rd.append(seg_data["LL_acc_z_kurt"])   
    Collect_rd.append(seg_data["LL_acc_z_peak1"])    
    Collect_rd.append(seg_data["LL_acc_z_peak2"])    
    Collect_rd.append(seg_data["LL_acc_z_peak3"])	   
    Collect_rd.append(seg_data["LL_acc_z_peak4"])   
    Collect_rd.append(seg_data["LL_acc_z_peak5"])    
    Collect_rd.append(seg_data["LL_acc_z_autocor0"])    
    Collect_rd.append(seg_data["LL_acc_z_autocor1"])    
    Collect_rd.append(seg_data["LL_acc_z_autocor2"])    
    Collect_rd.append(seg_data["LL_acc_z_autocor3"])
    Collect_rd.append(seg_data["LL_acc_z_autocor4"])    
    Collect_rd.append(seg_data["LL_acc_z_autocor5"])    
    Collect_rd.append(seg_data["LL_acc_z_autocor6"])    
    Collect_rd.append(seg_data["LL_acc_z_autocor7"])    
    Collect_rd.append(seg_data["LL_acc_z_autocor8"])
    Collect_rd.append(seg_data["LL_acc_z_autocor9"]) 
	
    labels_rd.append(seg_data["label"])
	 
labels_rd=np.array(labels_rd)
Collect_arr_rd=np.array(Collect_rd).reshape((2400,315))
Collect_norm_rd=preprocessing.normalize(Collect_arr_rd,norm="l2") #标准化
#PCA降维
#零均值化  
def zeroMean(dataMat):        
    meanVal=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值  
    newData=dataMat-meanVal  
    return newData,meanVal  

def percentage2n(eigVals,percentage):  #选择主成分个数
    sortArray=np.sort(eigVals)   #升序  
    sortArray=sortArray[-1::-1]  #逆转，即降序  
    arraySum=sum(sortArray)  
    tmpSum=0  
    num=0  
    for i in sortArray:  
        tmpSum+=i  
        num+=1  
        if tmpSum>=arraySum*percentage:  
            return num  
			
def pca(dataMat,percentage=0.99):  
    newData,meanVal=zeroMean(dataMat)  
    covMat=np.cov(newData,rowvar=0)   #求协方差矩阵,return ndarray；若rowvar为0，一行代表一个样本  
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的  
    n=percentage2n(eigVals,percentage)          #要达到percent的方差百分比，需要前n个特征向量  
    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序  
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标  
    n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量  
    lowDDataMat=newData*n_eigVect               #低维特征空间的数据  
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #重构数据  
    return lowDDataMat,reconMat,n  
	
lowDDataMat,reconMat,number=pca(Collect_norm_rd,percentage=0.9995)
print u"特征值数：",number #0.99=72 0.95=52,0.9=42,0.85=35
processed_data=lowDDataMat

#切分成10份
slice0=processed_data[0:240].copy()
slicelab0=labels_rd[0:240].copy()
#print len(slice0),"/n",len(slicelab0)
slice1=processed_data[240:480].copy()
slicelab1=labels_rd[240:480].copy()
#print len(slice1),"/n",len(slicelab1)
slice2=processed_data[480:720].copy()
slicelab2=labels_rd[480:720].copy()
#print len(slice2),"/n",len(slicelab2)
slice3=processed_data[720:960].copy()
slicelab3=labels_rd[720:960].copy()
#print len(slice3),"/n",len(slicelab3)
slice4=processed_data[960:1200].copy()
slicelab4=labels_rd[960:1200].copy()
#print len(slice4),"/n",len(slicelab4)
slice5=processed_data[1200:1440].copy()
slicelab5=labels_rd[1200:1440].copy()
#print len(slice5),"/n",len(slicelab5)
slice6=processed_data[1440:1680].copy()
slicelab6=labels_rd[1440:1680].copy()
#print len(slice6),"/n",len(slicelab6)
slice7=processed_data[1680:1920].copy()
slicelab7=labels_rd[1680:1920].copy()
#print len(slice7),"/n",len(slicelab7)
slice8=processed_data[1920:2160].copy()
slicelab8=labels_rd[1920:2160].copy()
#print len(slice8),"/n",len(slicelab8)
slice9=processed_data[2160:2400].copy()
slicelab9=labels_rd[2160:2400].copy()
#print len(slice9),"/n",len(slicelab9)

#合9留一 #分割训练集和测试集
data9_slice9=np.concatenate((slice0,slice1,slice2,slice3,slice4,slice5,slice6,slice7,slice8),axis=0)
data9_slicelab9=np.concatenate((slicelab0,slicelab1,slicelab2,slicelab3,slicelab4,slicelab5,slicelab6,slicelab7,slicelab8),axis=0)
data1_slice9=slice9
data1_slicelab9=slicelab9

data9_slice8=np.concatenate((slice0,slice1,slice2,slice3,slice4,slice5,slice6,slice7,slice9),axis=0)
data9_slicelab8=np.concatenate((slicelab0,slicelab1,slicelab2,slicelab3,slicelab4,slicelab5,slicelab6,slicelab7,slicelab9),axis=0)
data1_slice8=slice8
data1_slicelab8=slicelab8

data9_slice7=np.concatenate((slice0,slice1,slice2,slice3,slice4,slice5,slice6,slice9,slice8),axis=0)
data9_slicelab7=np.concatenate((slicelab0,slicelab1,slicelab2,slicelab3,slicelab4,slicelab5,slicelab6,slicelab9,slicelab8),axis=0)
data1_slice7=slice7
data1_slicelab7=slicelab7

data9_slice6=np.concatenate((slice0,slice1,slice2,slice3,slice4,slice5,slice9,slice7,slice8),axis=0)
data9_slicelab6=np.concatenate((slicelab0,slicelab1,slicelab2,slicelab3,slicelab4,slicelab5,slicelab9,slicelab7,slicelab8),axis=0)
data1_slice6=slice6
data1_slicelab6=slicelab6

data9_slice5=np.concatenate((slice0,slice1,slice2,slice3,slice4,slice9,slice6,slice7,slice8),axis=0)
data9_slicelab5=np.concatenate((slicelab0,slicelab1,slicelab2,slicelab3,slicelab4,slicelab9,slicelab6,slicelab7,slicelab8),axis=0)
data1_slice5=slice5
data1_slicelab5=slicelab5

data9_slice4=np.concatenate((slice0,slice1,slice2,slice3,slice9,slice5,slice6,slice7,slice8),axis=0)
data9_slicelab4=np.concatenate((slicelab0,slicelab1,slicelab2,slicelab3,slicelab9,slicelab5,slicelab6,slicelab7,slicelab8),axis=0)
data1_slice4=slice4
data1_slicelab4=slicelab4

data9_slice3=np.concatenate((slice0,slice1,slice2,slice9,slice4,slice5,slice6,slice7,slice8),axis=0)
data9_slicelab3=np.concatenate((slicelab0,slicelab1,slicelab2,slicelab9,slicelab4,slicelab5,slicelab6,slicelab7,slicelab8),axis=0)
data1_slice3=slice3
data1_slicelab3=slicelab3

data9_slice2=np.concatenate((slice0,slice1,slice9,slice3,slice4,slice5,slice6,slice7,slice8),axis=0)
data9_slicelab2=np.concatenate((slicelab0,slicelab1,slicelab9,slicelab3,slicelab4,slicelab5,slicelab6,slicelab7,slicelab8),axis=0)
data1_slice2=slice2
data1_slicelab2=slicelab2

data9_slice1=np.concatenate((slice0,slice9,slice2,slice3,slice4,slice5,slice6,slice7,slice8),axis=0)
data9_slicelab1=np.concatenate((slicelab0,slicelab9,slicelab2,slicelab3,slicelab4,slicelab5,slicelab6,slicelab7,slicelab8),axis=0)
data1_slice1=slice1
data1_slicelab1=slicelab1

data9_slice0=np.concatenate((slice9,slice1,slice2,slice3,slice4,slice5,slice6,slice7,slice8),axis=0)
data9_slicelab0=np.concatenate((slicelab9,slicelab1,slicelab2,slicelab3,slicelab4,slicelab5,slicelab6,slicelab7,slicelab8),axis=0)
data1_slice0=slice0
data1_slicelab0=slicelab0

train_d0,test_d0,train_lab0,test_lab0=data9_slice9,data1_slice9,data9_slicelab9,data1_slicelab9
train_d1,test_d1,train_lab1,test_lab1=data9_slice8,data1_slice8,data9_slicelab8,data1_slicelab8
train_d2,test_d2,train_lab2,test_lab2=data9_slice7,data1_slice7,data9_slicelab7,data1_slicelab7
train_d3,test_d3,train_lab3,test_lab3=data9_slice6,data1_slice6,data9_slicelab6,data1_slicelab6
train_d4,test_d4,train_lab4,test_lab4=data9_slice5,data1_slice5,data9_slicelab5,data1_slicelab5
train_d5,test_d5,train_lab5,test_lab5=data9_slice4,data1_slice4,data9_slicelab4,data1_slicelab4
train_d6,test_d6,train_lab6,test_lab6=data9_slice3,data1_slice3,data9_slicelab3,data1_slicelab3
train_d7,test_d7,train_lab7,test_lab7=data9_slice2,data1_slice2,data9_slicelab2,data1_slicelab2
train_d8,test_d8,train_lab8,test_lab8=data9_slice1,data1_slice1,data9_slicelab1,data1_slicelab1
train_d9,test_d9,train_lab9,test_lab9=data9_slice0,data1_slice0,data9_slicelab0,data1_slicelab0

#print train_d0.shape,'/n',train_lab0.shape,'/n',test_d0.shape,'/n',test_lab0.shape
#print train_d1.shape,'/n',train_lab1.shape,'/n',test_d1.shape,'/n',test_lab1.shape
#print train_d2.shape,'/n',train_lab2.shape,'/n',test_d2.shape,'/n',test_lab2.shape
#print train_d3.shape,'/n',train_lab3.shape,'/n',test_d3.shape,'/n',test_lab3.shape
#print train_d4.shape,'/n',train_lab4.shape,'/n',test_d4.shape,'/n',test_lab4.shape
#print train_d5.shape,'/n',train_lab5.shape,'/n',test_d5.shape,'/n',test_lab5.shape
#print train_d6.shape,'/n',train_lab6.shape,'/n',test_d6.shape,'/n',test_lab6.shape
#print train_d7.shape,'/n',train_lab7.shape,'/n',test_d7.shape,'/n',test_lab7.shape
#print train_d8.shape,'/n',train_lab8.shape,'/n',test_d8.shape,'/n',test_lab8.shape
#print train_d9.shape,'/n',train_lab9.shape,'/n',test_d9.shape,'/n',test_lab9.shape

#plt.figure()
#X=train_d0
#gmm = GMM(4,X)
# train GMM0
#gmm.gmm(X,4)
#index=predictIndx(gmm,X)
    
#plt.scatter(X[index==0][:,0],X[index==0][:,1],s=150,c=u'r',marker=u'o')
#plt.scatter(X[index==1][:,0],X[index==1][:,1],s=150,c=u'b',marker=u'o')
#plt.scatter(X[index==2][:,0],X[index==2][:,1],s=150,c=u'y',marker=u'o')
#plt.scatter(X[index==3][:,0],X[index==3][:,1],s=60,c=u'g',marker=u'o')
#plt.title(u'k=4时的聚类结果',fontsize=15)
#plt.savefig(r"C:\test\k=4.png",dpi=800) #绘制当前图像
#plt.show()
def train_and_test(train_data,test_data,train_label,test_label,number):#训练集和测试集的准确率
#每一类单独训练
    X0train=[]
    X1train=[]
    X2train=[]
    X3train=[]
    X4train=[]
    for index,i in enumerate(train_label):
        if i==0:
            X0train.append(train_data[index])
        elif i==1:
            X1train.append(train_data[index])
        elif i==2:
            X2train.append(train_data[index])
        elif i==3:
            X3train.append(train_data[index])
        elif i==4:
            X4train.append(train_data[index])
    def transform2D(X,n):
        X=np.array(X).tolist()
        X=np.array(X).reshape((-1,n))
        return X
    X0train=np.mat(transform2D(X0train,number))
    X1train=np.mat(transform2D(X1train,number))
    X2train=np.mat(transform2D(X2train,number))
    X3train=np.mat(transform2D(X3train,number))
    X4train=np.mat(transform2D(X4train,number))
#print X0train.shape
#print X1train.shape
#print X2train.shape
#print X3train.shape
#print X4train.shape
#print train_data.shape

#training
    k0=1 #2的结果好
    k1=1
    k2=1
    k3=1
    k4=1

    gmm0 = GMM(k0,X0train)
    # train GMM0
    gmm0.gmm(X0train,k0)
#    print "reslut:",gmm0.gmm(X0train,k0)
    gmm1 = GMM(k1,X1train)
    gmm1.gmm(X1train,k1)

    gmm2 = GMM(k2,X2train)
    gmm2.gmm(X2train,k2)

    gmm3 = GMM(k3,X3train)
    gmm3.gmm(X3train,k3)

    gmm4 = GMM(k4,X4train)
    gmm4.gmm(X4train,k4)

    #画图
    
#    index0=predictIndx(gmm0,X0train)
#    print len(index0)
#    index1=predictIndx(gmm1,X1train)
#    print len(index1)
#    index2=predictIndx(gmm2,X2train)
#    print len(index2)
#    index3=predictIndx(gmm3,X3train)
#    print len(index3)
#    index4=predictIndx(gmm4,X4train)
#    print len(index4)
#    plt.figure()
#    X=train_data
#    plt.scatter(X0train[index0==0][:,0],X0train[index0==0][:,1],s=60,c=u'r',marker=u'o')
#    plt.scatter(X0train[index0==1][:,0],X0train[index0==1][:,1],s=60,c=u'b',marker=u'o')
     
#    plt.scatter(X1train[index1==0][:,0],X1train[index1==0][:,1],s=60,c=u'saddlebrown',marker=u'o')
#    plt.scatter(X1train[index1==1][:,0],X1train[index1==1][:,1],s=60,c=u'palegreen',marker=u'o')
	
#    plt.scatter(X2train[index2==0][:,0],X2train[index2==0][:,1],s=60,c=u'y',marker=u'o')
#    plt.scatter(X2train[index2==1][:,0],X2train[index2==1][:,1],s=60,c=u'orange',marker=u'o')
	
#    plt.scatter(X3train[index3==0][:,0],X3train[index3==0][:,1],s=60,c=u'lawngreen',marker=u'o')
#    plt.scatter(X3train[index3==1][:,0],X3train[index3==1][:,1],s=60,c=u'purple',marker=u'o')
	
#    plt.scatter(X4train[index4==0][:,0],X4train[index4==0][:,1],s=60,c=u'pink',marker=u'o')
#    plt.scatter(X4train[index4==1][:,0],X4train[index4==1][:,1],s=60,c=u'cyan',marker=u'o')
#    plt.title(u"训练数据的聚类结果",fontsize=15)
#    plt.savefig(r"C:\test\10classorg.png",dpi=800) #绘制当前图像
#    plt.show()
	
	
    re_train=generateResult(gmm0, gmm1,gmm2,gmm3,gmm4, train_data)
#    print re_train,"/n",len(re_train)
    count=0.0
    for index,i in enumerate(re_train):
        if i==train_label[index]:
            count+=1
    train_accuracy=count/len(train_label)
#    print "train_accuracy:",train_accuracy
	
#testing

    re_test=generateResult(gmm0, gmm1,gmm2,gmm3,gmm4, test_data)
    #    print re_train,"/n",len(re_train)
    count=0.0
    for index,i in enumerate(re_test):
        if i==test_label[index]:
            count+=1
    test_accuracy=count/len(test_label)
#    print "test_accuracy:",test_accuracy
    return train_accuracy,test_accuracy
	
sum_train_acc=[]        #求10份集的准确率
sum_test_acc=[]
tracc0,tesacc0=train_and_test(train_d0,test_d0,train_lab0,test_lab0,number)
sum_train_acc.append(tracc0)
sum_test_acc.append(tesacc0)

tracc1,tesacc1=train_and_test(train_d1,test_d1,train_lab1,test_lab1,number)
sum_train_acc.append(tracc1)
sum_test_acc.append(tesacc1)

tracc2,tesacc2=train_and_test(train_d2,test_d2,train_lab2,test_lab2,number)
sum_train_acc.append(tracc2)
sum_test_acc.append(tesacc2)

tracc3,tesacc3=train_and_test(train_d3,test_d3,train_lab3,test_lab3,number)
sum_train_acc.append(tracc3)
sum_test_acc.append(tesacc3)

tracc4,tesacc4=train_and_test(train_d4,test_d4,train_lab4,test_lab4,number)
sum_train_acc.append(tracc4)
sum_test_acc.append(tesacc4)

tracc5,tesacc5=train_and_test(train_d5,test_d5,train_lab5,test_lab5,number)
sum_train_acc.append(tracc5)
sum_test_acc.append(tesacc5)

tracc6,tesacc6=train_and_test(train_d6,test_d6,train_lab6,test_lab6,number)
sum_train_acc.append(tracc6)
sum_test_acc.append(tesacc6)

tracc7,tesacc7=train_and_test(train_d7,test_d7,train_lab7,test_lab7,number)
sum_train_acc.append(tracc7)
sum_test_acc.append(tesacc7)

tracc8,tesacc8=train_and_test(train_d8,test_d8,train_lab8,test_lab8,number)
sum_train_acc.append(tracc8)
sum_test_acc.append(tesacc8)

tracc9,tesacc9=train_and_test(train_d9,test_d9,train_lab9,test_lab9,number)
sum_train_acc.append(tracc9)
sum_test_acc.append(tesacc9)
print u"k=1 的聚类结果"
s1=0
for i in sum_train_acc:
    s1+=i
print u"train data的准确率:",s1/10

s2=0
for i in sum_test_acc:
    s2+=i
print u"test data的准确率：",s2/10
