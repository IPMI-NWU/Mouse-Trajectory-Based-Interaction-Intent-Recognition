'''
划分基础事件
'''
import pandas as pd
import move_event_process
import numpy as np
import res_tool
from pandas.core.frame import DataFrame
import os

mouse_type=res_tool.mouse_type
basic_event=res_tool.basic_event
predict=[]


def point_event_judge(region,mouse_type,mouse_object):
    event_type=0  #初始化为0
    if(region==-1 and mouse_type==1 and mouse_object>-1):
        event_type=8  #检测button
    if(region in [1,2,3,4,5,6] and mouse_type==1):  #button事件
        event_type=region
    if(mouse_type==7):  #切换影像
        event_type=9
    if(mouse_type==8):  #切换检测结果
        event_type=13
    if (mouse_type == 9):  # 切换辅助检测工具类型
        event_type = 10
    if(mouse_type==2 and region in [14,15]):  #进入图像详情
        event_type=16
    if(mouse_type==2 and region==17): #退出图像详情
        event_type = 17
    if(mouse_type==5 and region in [17,18]):  #放大图像
        event_type=18
    if (mouse_type == 6 and region in [17, 18]):  # 缩小图像
        event_type = 19
    if (mouse_type == 3 and region==17):  # 移动图像
        event_type = 20


    return event_type

def move_event_judge(move_event):
    region=move_event.iloc[0,1]
    type=0
    predict=move_event_process.classifier1(move_event)
    if(predict==0):   #1表示无意义事件,0表示浏览事件
        if(region==7):
            type=7
        elif (region == 12): #诊断结果列表区域
            type = 12
        elif (region == 13):  #诊断报告区域
            type = 11
        elif (region == 14):
            type = 14
        elif (region == 15):
            type = 15
        elif (region == 17):
            type = 21
    return type



def event_type_judge(segment,snippet):
    # 先判断是否存在有意义的snippet
    event_type=0
    for value in list(snippet.items()):
        value = value[1] #value[0]是字典编号，无意义
        region=value.iloc[0,1]  #因为每个snippet的region和mouse_type都一样
                                 #就直接用每个snippet第一行的数据来表示该snippet
        mouse_type=value.iloc[0,4]
        mouse_object=value.iloc[0,5]
        event_type=event_type+point_event_judge(region,mouse_type,mouse_object)
    if(event_type>0):  #一个segment中的snippet存在有意义的点事件
                       #此时，对所有snippet依次判断
        for value in list(snippet.items()):
            value = value[1]  # value[0]是字典编号，无意义
            region = value.iloc[0,1]  # 因为每个snippet的region和mouse_type都一样
            # 就直接用每个snippet第一行的数据来表示该snippet
            mouse_type = value.iloc[0,4]
            mouse_object = value.iloc[0, 5]
            type=point_event_judge(region, mouse_type,mouse_object)
            t_start=value.iloc[0,0]
            t_end=value.iloc[-1,0]
            if(type>0):
                predict.append(['snippet',type,basic_event[int(type)],t_start, t_end,value.iloc[0,5],
                                value.iloc[0, 6],value.iloc[0,7],value.iloc[0,8],value.iloc[0,9],
                                value.iloc[0, 10],value.iloc[0,11],value.iloc[0,12],value.iloc[0,13],
                                value.iloc[-1,8],value.iloc[-1,9],value.iloc[-1,10],value.iloc[-1,11],value.iloc[-1,12],value.iloc[-1,13]])
                print('snippet 点事件:%s t_start:%d t_end:%d %d'%(basic_event[int(type)],t_start,t_end,value.iloc[0,5]))
            else:  #对于移动事件，用机器学习模型判断
                type=move_event_judge(value)
                print('snippet 移动事件:%s t_start:%d t_end:%d %d'%(basic_event[int(type)],t_start,t_end,value.iloc[0,5]))
                predict.append(['snippet', type, basic_event[int(type)], t_start, t_end, value.iloc[0, 5],
                                value.iloc[0, 6], value.iloc[0, 7], value.iloc[0, 8], value.iloc[0, 9],
                                value.iloc[0, 10], value.iloc[0, 11],value.iloc[0, 12], value.iloc[0, 13],
                                value.iloc[-1, 8], value.iloc[-1, 9], value.iloc[-1, 10], value.iloc[-1, 11],
                                 value.iloc[-1, 12], value.iloc[-1, 13]])
    else:  #一个segment中的所有snippet都不存在有意义的点事件
           #此时忽略点事件类型，用机器学习模型判断
        type=move_event_judge(segment)
        t_start = segment.iloc[0, 0]
        t_end = segment.iloc[-1, 0]
        print('segment 移动事件:%s t_start:%d t_end:%d %d'%(basic_event[int(type)],t_start,t_end,value.iloc[0,5]))
        predict.append(['segment', type, basic_event[int(type)], t_start, t_end, value.iloc[0, 5],
                           value.iloc[0, 6], value.iloc[0, 7], value.iloc[0, 8], value.iloc[0, 9],
                           value.iloc[0, 10], value.iloc[0, 11], value.iloc[0, 12],value.iloc[0, 13],
                           value.iloc[-1, 8], value.iloc[-1, 9], value.iloc[-1, 10], value.iloc[-1, 11],
                           value.iloc[-1, 12], value.iloc[-1, 13]])


def data_segmatation_region(data):
# 对数据进行分段
# 将连续的数据以region划分为多个连续不等长的segment
    segment={}
    j=0
    num=0
    for i in range(1,len(data)):
        if(data.iloc[i,1]==data.iloc[i-1,1]):
            continue
        else:
            segment[num]=data.iloc[j:i,:]
            num+=1
            j=i
    segment[num]=data.iloc[j:i+1,:]
    for key, value in segment.items():
        data_segmatation_mousetype(value)
    #return segment

def data_segmatation_mousetype(segment):
# 对segment再次进行划分
# 将连续的数据以mousetype划分为多个连续不等长的snippet
    snippet = {}
    j = 0
    num = 0
    if(len(segment)==1):
        snippet[num]=segment
    else:
        for i in range(1, len(segment)):
            if (segment.iloc[i, 4] == segment.iloc[i - 1, 4]):
                continue
            else:
                snippet[num] = segment.iloc[j:i, :]
                num += 1
                j = i
        snippet[num] = segment.iloc[j:i + 1, :]
    event_type_judge(segment,snippet)



if __name__ == '__main__':
    path = '1_preprocess/with_label/'
    files = os.listdir(path)
    for file_name in files:
        address = path + file_name
        name = os.path.splitext(file_name)[0]
        data = pd.read_csv(address)
        data['basic event'] = 0
        predict=[]
        data_segmatation_region(data)
        m = DataFrame(predict, columns=['type', 'coarseNo', 'coarseName', 'startT', 'endT', 'mouse_object',
                                        'view_parm', 'result_parm', 'imageW', 'imageH', 'x_start', 'x_end',
                                        'y_start', 'y_end', 'end_imageW', 'end_iamgeH', 'end_x_start', 'end_x_end',
                                        'end_y_start', 'end_y_end'])
        m.to_csv('2_coarse_event/original/with_label/new/%s.csv'%name, index=False, encoding="utf_8_sig")



