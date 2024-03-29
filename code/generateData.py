#!/usr/bin/python
#coding:utf-8
import numpy as np
from gmm3 import *

def generateDataCat(filename, cat):
#	"""
#	return a N by D matrix X 
#	"""
    f = open(filename,'r')
    data = []

    for line in f:
        line = line.replace('\n','')
        decode = map(float,line.split())
        
        if decode[2] - 1 == cat:
            data.append([decode[0],decode[1]])
			
    return np.mat(data)
	
def generateData(filename):
    f = open(filename,'r')
    data = []

    for line in f:
        line = line.replace('\n','')
        decode = map(float,line.split())
        data.append([decode[0],decode[1]])

    return np.mat(data)
	
def generateAns(filename):
    f = open(filename,'r')
    ans = []

    for line in f:
        line = line.replace('\n','')
        decode = map(float,line.split())
        ans.append(decode[2])
	
    return ans
	
def generateResult(gmmModel0, gmmModel1,gmmModel2,gmmModel3,gmmModel4, X):   #max 的推广
    re0 = predictValue(gmmModel0,X)
    re1 = predictValue(gmmModel1,X)
    re2 = predictValue(gmmModel2,X)
    re3 = predictValue(gmmModel3,X)
    re4 = predictValue(gmmModel4,X)
    re = []
    for index,i in enumerate(re0):
        L=[i,re1[index],re2[index],re3[index],re4[index]]  
        indice=np.argmax(np.array(L))       #下标
#	bre = re0 > re1
#        re = []
        re.append(indice)
#	for b in bre:
#        	if b:
#                	re.append(0)
#        	else:
#                	re.append(1)
    return re

def generateResult1(gmmModel0, gmmModel1, X):
    re0 = predictValue(gmmModel0,X)
    re1 = predictValue(gmmModel1,X)
    
    bre = re0 > re1
    re = []
    
    for b in bre:
        if b:
            re.append(0)
        else:
            re.append(1)
    return re
if __name__ == "__main__":
    X0 = generateDataCat(r'C:/test/train.txt',0) 
#    print "X0: ", X0
    X = generateData(r'C:/test/train.txt')
#    print "X: ", X
    ans=generateAns(r'C:/test/train.txt')
    print "ans",ans