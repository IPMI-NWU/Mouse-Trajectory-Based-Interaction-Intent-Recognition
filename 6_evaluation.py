import numpy as np
import json
import os
import pandas as pd
from pandas.core.frame import DataFrame
tThreshold = 0.8

point_event=['打开文件','打开文件夹','保存诊断结果到Execl','打印诊断报告','帮助','轨迹录制','使用辅助诊断工具','切换图像','上一张图像','下一张图像','选择辅助诊断工具类型',
             '选择辅助诊断结果','进入图像详情模式','退出图像详情模式']
continous_event=['浏览辅助诊断工具箱','阅读诊断报告','浏览辅助诊断结果列表','浏览辅助诊断图像','放大图像','缩小图像','移动图像','浏览图像详情','浏览原图']

#fp那些计算有点问题，可以直接舍去
evaluation_result=[]


class Proposal:
    def __init__(self,Serial_number,name,st,et,match=0,tiou=0):
        self.Serial_number=Serial_number
        self.name = name
        self.st=st
        self.et=et
        self.match=match
        self.tiou=tiou

class Groundtruth:
    def __init__(self, Serial_number,name,start_time, end_time,type=None,match=0,tiou=0):
        self.Serial_number = Serial_number
        self.name=name
        self.start_time = start_time
        self.end_time = end_time
        self.match = match
        self.tiou=tiou
        self.type=type  #0代表连续型事件，1代表点事件

def load_groundtruth(filename):
    f = open('label/' + filename+'.json', 'r', encoding='utf-8')
    info_data = json.load(f)
    name = os.path.splitext(file_name)[0]
    data=info_data[name]['information']  #groundtruth，列表格式
    gt=[]
    for i in range(len(data)):
        label_name = data[i]['label']  # groundtruth类型名
        label_st = data[i]['segment'][0]  # 开始时间
        label_et = data[i]['segment'][1]  # 结束时间
        ft_data = Groundtruth(i+1,label_name, label_st, label_et)  #Groundtruth对象初始化
        gt.append(ft_data)
    return gt


def load_predict(filename):
    path='2_coarse_event/postprocessed/new/'
    df=pd.read_csv(path+filename+'.csv')
    proposal = []
    for i in range(len(df)):
        if df.iloc[i,1]=='选择辅助检测工具类型':
            df.iloc[i,1]='选择辅助诊断工具类型'
        if df.iloc[i,1] in continous_event:
            if df.iloc[i,3]-df.iloc[i,2]>=100:  #忽略duration<500的检测事件
                proposal_data = Proposal(i + 1, df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3])
                proposal.append(proposal_data)
        else:
            proposal_data = Proposal(i + 1, df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3])
            proposal.append(proposal_data)

    return proposal


def event_type(name):
    # 判断一个事件是点事件还是连续事件
    if(name in point_event):
        return True  #点事件返回true
    else:
        return False  #连续型事件返回false

def tIou(t_start_1,t_end_1,t_start_2,t_end_2):
    t_start_1=float(t_start_1)
    t_start_2=float(t_start_2)
    t_end_1=float(t_end_1)
    t_end_2 = float(t_end_2)
    t1=np.maximum(t_start_1,t_start_2)
    t2=np.minimum(t_end_1,t_end_2)
    intersection = (t2 - t1).clip(0)
    proposal_time=t_end_1-t_start_1
    # union = (t_end_1 - t_start_1) \
    #                  + (t_end_2 - t_start_2) - intersection
    # tiou = intersection.astype(float) /union
    tiou=intersection/proposal_time
    return tiou

def count(label,predict,tThreshold):
    for i in range(len(predict)):
        for j in range(len(label)):
            if predict[i].name==label[j].name:  #类型正确
                if predict[i].name not in continous_event:
                    if predict[i].st>=label[j].start_time or predict[i].et<=label[j].end_time:
                        predict[i].match = 1
                        label[j].tiou = 1
                        label[j].match = 1
                        predict[i].tiou=1
                        continue

                else:
                    tiou=tIou(predict[i].st,predict[i].et,label[j].start_time,label[j].end_time)
                    if tiou>=tThreshold:
                        predict[i].match=1
                        label[j].match=1
                        predict[i].tiou=tiou
                        label[j].tiou=max(tiou,label[j].tiou)
    return label,predict


def evaluation(label,predict,tThreshold,name):
    label, predict = count(label, predict, tThreshold)
    proposal_match=0
    for i in predict:
        proposal_match+=i.match
    gt_match=0
    for i in label:
        gt_match+=i.match
    gt_num=len(label)  #groundtruth中事件的个数
    proposal_num=len(predict)
    fn=gt_num-gt_match  #有多少gt没有被检测出来
    fp=proposal_num-proposal_match  #检测的proposal中有多少错误了
    tp=gt_match
    recall=tp/(tp+fn)
    precision=tp/(tp+fp)
    #调和均值F=3*P*R/(2*P+R)
    f1_score=3*precision*recall/(2*precision+recall)
    # print('TP:%d  FP:%d  FN:%d'%(tp,fp,fn))
    # print('recall:%f  precision:%f  f1_score:%f'%(recall,precision,f1_score))
    evaluation_result.append([name,gt_num,proposal_num,tp,fp,fn,recall,precision,f1_score])



if __name__ == '__main__':
    path = 'label/'
    files = os.listdir(path)
    for file_name in files:
        name = os.path.splitext(file_name)[0]
        label = load_groundtruth(name)  # gt真实行为个数
        predict = load_predict(name)  # 预测行为个数
        evaluation(label, predict, tThreshold,name)
    m = DataFrame(evaluation_result, columns=['file_name','gt_num','proposal_num','tp','fp','fn',
                                              'recall','precision','f1_score'])
    m.to_csv('evaluation_result/tiou_0.8.csv', index=False)



