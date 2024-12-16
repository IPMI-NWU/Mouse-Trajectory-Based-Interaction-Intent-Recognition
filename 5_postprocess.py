import os
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

time_threshold=10000 #用来对预测结果做后处理的，时间间隔阈值
noResponseCoarse=[7,11,12,14,15,21]

def postprocess(df,filename):
    '''
    对预测结果做后处理
    合并相邻的类型相同且时间间隔小于某阈值的预测行为事件
    '''
    data_new = []
    data_final=[]
    i=0
    while i<len(df)-1:
        st = df.iloc[i,3] #start_time
        et = df.iloc[i,4] #end_time
        for j in range(i+1,len(df)):
            interval_time = float(df.iloc[j,3]) - float(et)
            if(df.iloc[i,1]!=df.iloc[j,1] or interval_time>time_threshold): #不用合并
                df.iloc[i, 3] = st
                df.iloc[i, 4] = et
                data_new.append(df.iloc[i,1:])
                i=j
                break
            else:
                if(i==len(df)-2):
                    df.iloc[i,3]=st
                    df.iloc[i,4]=et
                    data_new.append(df.iloc[i,1:])
                else:
                    et=df.iloc[j,4]
            i=j
    interval_time = float(df.iloc[-1,3]) - float(et)
    if (df.iloc[i,1] != data_new[-1][0] or interval_time > time_threshold):
        data_new[-1][3] = df.iloc[i, 4]
    else:
        data_new.append(df.iloc[i, 1:19])
    for i in range(len(data_new)):
        if (data_new[i][0] in noResponseCoarse):
            if(data_new[i][3] - data_new[i][2] >= 1000):
                print(data_new[i][1],data_new[i][3] - data_new[i][2])
                data_final.append(data_new[i])
        else:
            data_final.append(data_new[i])
    data_final=DataFrame(data_final)
    data_final.to_csv('2_coarse_event/postprocessed/new/%s.csv' % filename, index=False, encoding="utf_8_sig")


if __name__ == '__main__':
    path = '2_coarse_event/original/with_label/new/'
    files = os.listdir(path)
    for file_name in files:
        #读取该路径下的所有.npy文件，进行进一步后处理
        predict = []
        address = path + file_name
        name = os.path.splitext(file_name)[0]
        df=pd.read_csv(address)
        df = df.drop(df[df.coarseNo == 0].index)  #删除所有背景事件
        postprocess(df,name)


#删除所有持续时间小于1秒的非系统响应行为