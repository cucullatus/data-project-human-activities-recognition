#-*— coding: utf-8 -*-
#!/usr/bin/python
#文件夹下有多个文件嵌套
import os
import numpy as np
import cPickle as cp

def read_data_path(path):     #读路径
    filepath=[]
    for root,dirt,filename in os.walk(path): #遍历当前文件夹
        for name in filename:                 #遍历文件
            filepath.append(os.path.join(root,name))#拼接路径下所有文件,list格式
#    parent_path=os.path.dirname(filepath)
    return filepath

def read_data(filepath):   #读路径文件中数据    
    file_dict=[]
    for filename in filepath: #遍历文件，取数据
        file_dict.append(np.loadtxt(filename,delimiter=","))
    
    file_data=[]
    for i in file_dict:
        for elements in i.flat:
            
            file_data.append(elements)
	
			
    file_data=np.array(file_data).reshape((-1,45))       #list 转array，中间没有","
    f=open(r"C:\test\data05.pkl","wb")
    cp.dump(file_data,f)
    return len(file_data)
 #   return file_dict
if __name__=='__main__':
    path="C:\data"           #输入路径
    filepath=read_data_path(path)
    
    read_data(filepath)
 #   print read_data(filepath)
 
