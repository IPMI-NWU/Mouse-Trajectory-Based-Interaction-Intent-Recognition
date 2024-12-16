import pandas as pd
import numpy as np
import joblib
from collections import Counter
import math

# 训练好的机器学习模型
browseImage=joblib.load(filename='../utils/DT_Image.pkl')
browseReport=joblib.load(filename='../utils/DT_Report.pkl')
browseToolBox=joblib.load(filename='../utils/RFC_ReadToolBox.pkl')
browseResultList=joblib.load(filename='../utils/LR_ResultList.pkl')


# 一段轨迹距离计算
def cal_featureDist(data):
    totalDist, totalX, totalY = [0, 0, 0]
    maxDist, maxX, maxY = [0, 0, 0]
    for i in range(len(data) - 1):
        x0 = data.iloc[i, 2]
        y0 = data.iloc[i, 3]
        x1 = data.iloc[i + 1, 2]
        y1 = data.iloc[i + 1, 3]
        r = pow(pow(x1 - x0, 2) + pow(y1 - y0, 2), 0.5)
        x = abs(x1 - x0)
        y = abs(y1 - y0)
        if r > maxDist: maxDist = r
        if x > maxX: maxX = x
        if y > maxY: maxY = y
        totalDist = totalDist + r
        totalX = totalX + x
        totalY = totalY + y
    avgDist = totalDist / len(data)
    avgX = totalX / (len(data) - 1)
    avgY = totalY / (len(data) - 1)
    sdDist, sdX, sdY = distanceSD(data, avgDist, avgX, avgY)
    # 总距离、最大距离、平均距离、距离标准差、X轴...、Y轴... （12个特征）
    # 滑动路径平滑度
    Str = pow(pow(data.iloc[0, 2] - data.iloc[-1, 2], 2) + pow(data.iloc[0, 3] - data.iloc[-1, 3], 2), 0.5) / totalDist
    return totalDist, maxDist, avgDist, sdDist, totalX, maxX, avgX, sdX, totalY, maxY, avgY, sdY, Str


def distanceSD(data, avgDist, avgX, avgY):
    total, totalX, totalY = [0, 0, 0]
    for i in range(len(data) - 1):
        x0 = data.iloc[i, 2]
        y0 = data.iloc[i, 3]
        x1 = data.iloc[i + 1, 2]
        y1 = data.iloc[i + 1, 3]
        r = pow(pow(x1 - x0, 2) + pow(y1 - y0, 2), 0.5)
        x = abs(x1 - x0)
        y = abs(y1 - y0)
        total = total + pow(avgDist - r, 2)
        totalX = totalX + pow(avgX - x, 2)
        totalY = totalY + pow(avgY - y, 2)
    return pow(total / (len(data) - 1), 0.5), pow(totalX / (len(data) - 1), 0.5), pow(totalY / (len(data) - 1), 0.5)


def x_Offset(data):  # 输入data的size为n×15
    # x轴偏移量
    x_offset = abs(data.iloc[-1, 2] - data.iloc[0, 2])
    return x_offset


def y_Offset(data):
    # y轴偏移量
    y_offset = abs(data.iloc[-1, 3] - data.iloc[0, 3])
    return y_offset


def cal_ang(point_1, point_2, point_3):
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
    a = math.sqrt(
        (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1]))
    b = math.sqrt(
        (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1]))
    c = math.sqrt(
        (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1]))
    b1 = (b * b - a * a - c * c) / (-2 * a * c)
    b1 = int(b1 * 1000) / 1000
    B = math.degrees(math.acos(b1))
    return 180 - B


def avg_angle(data):
    # 平均角度转变率
    angle = 0
    num = len(data) - 2
    for i in range(num):
        angle_new = cal_ang((data.iloc[i, 2], data.iloc[i, 3]),
                            (data.iloc[i + 1, 2], data.iloc[i + 1, 3]),
                            (data.iloc[i + 2, 2], data.iloc[i + 2, 3]))
        angle = angle + angle_new
    return angle / num

#一段轨迹的速度平均值计算:对所有轨迹段的速度和求平均，而不是总距离/总时间
def cal_featureSpeed(data):
    num=len(data)-1
    avgV,avgX,avgY=[0,0,0]
    maxV,maxX,maxY=[0,0,0]
    minV=float('inf')
    for i in range(num):
        x0=data.iloc[i,2] #x坐标
        x1=data.iloc[i+1,2]
        y0=data.iloc[i,3] #y坐标
        y1=data.iloc[i+1,3]
        d=pow(pow(x1-x0,2)+pow(y1-y0,2),0.5)
        x=abs(x1-x0)
        y=abs(y1-y0)
        t=data.iloc[i+1,0]-data.iloc[i,0]+1
        v=d/t*1000
        vx=x/t*1000
        vy=y/t*1000
        if v>maxV:maxV=v
        if v<minV:minV=v
        if vx>maxX:maxX=vx
        if vy>maxY:maxY=vy
        avgV=avgV+v
        avgX=avgX+vx
        avgY=avgY+vy
    avgV=avgV/num
    avgX=avgX/num
    avgY=avgY/num
    sdV,sdX,sdY=speedSD(data,avgV,avgX,avgY)
    #最小速度、最大速度、平均速度，速度标准差、X轴...、Y轴...（10个特征）
    return minV,maxV,avgV,sdV,maxX,avgX,sdX,maxY,avgY,sdY

#一段轨迹的速度标准差计算
def speedSD(data,avgV,avgX,avgY):
    sdV,sdX,sdY=[0,0,0]
    num=len(data)-1
    for i in range(num):
        x0=data.iloc[i,2] #x坐标
        x1=data.iloc[i+1,2]
        y0=data.iloc[i,3] #y坐标
        y1=data.iloc[i+1,3]
        d=pow(pow(x1-x0,2)+pow(y1-y0,2),0.5)
        x=abs(x1-x0)
        y=abs(y1-y0)
        t=data.iloc[i+1,0]-data.iloc[i,0]+1
        v=d/t*1000
        vx=x/t*1000
        vy=y/t*1000
        sdV=sdV+pow(v-avgV,2)
        sdX=sdX+pow(vx-avgX,2)
        sdY=sdY+pow(vy-avgY,2)
    sdV=pow(sdV/num,0.5)
    sdX=pow(sdX/num,0.5)
    sdY=pow(sdY/num,0.5)
    return sdV,sdX,sdY

#一段轨迹的加速度平均值计算
def avg_acc(data):
    num=len(data)-2
    avg_a=0
    for i in range(num):
        x0=data.iloc[i,2] #x坐标
        x1=data.iloc[i+1,2]
        x2=data.iloc[i+2,2]
        y0=data.iloc[i,3] #y坐标
        y1=data.iloc[i+1,3]
        y2=data.iloc[i+2,3]
        d1=pow(pow(x1-x0,2)+pow(y1-y0,2),0.5)
        d2=pow(pow(x2-x1,2)+pow(y2-y1,2),0.5)
        t1=data.iloc[i+1,0]-data.iloc[i,0]+1
        t2=data.iloc[i+2,0]-data.iloc[i+1,0]+1
        v1=d1/t1*1000
        v2=d2/t2*1000
        a=abs((v2-v1)/t1)  #求个绝对值，不然加速度有正负
        avg_a=avg_a+a
    avg_a=avg_a/num
    return avg_a

def std_acc(data):
    num=len(data)-2
    std_a=0
    avg_a=avg_acc(data)
    for i in range(num):
        x0=data.iloc[i,2] #x坐标
        x1=data.iloc[i+1,2]
        x2=data.iloc[i+2,2]
        y0=data.iloc[i,3] #y坐标
        y1=data.iloc[i+1,3]
        y2=data.iloc[i+2,3]
        d1=pow(pow(x1-x0,2)+pow(y1-y0,2),0.5)
        d2=pow(pow(x2-x1,2)+pow(y2-y1,2),0.5)
        t1=data.iloc[i+1,0]-data.iloc[i,0]+1
        t2=data.iloc[i+2,0]-data.iloc[i+1,0]+1
        v1=d1/t1*1000
        v2=d2/t2*1000
        a=(v2-v1)/t1
        std_a=std_a+pow(a-avg_a,2)
    std_a=pow(std_a/num,0.5)
    return std_a


def detect_staypoints(timeThreh, distThreh, mat):  # 停止点检测算法
    i = 0
    pointNum = len(mat)
    stayPoint = []
    while i < pointNum:
        j = i + 1
        token = 0
        while j < pointNum:
            dist = Distance(i, j, mat)
            if dist > distThreh:
                intervalT = mat.iloc[j, 0] - mat.iloc[i, 0]
                if intervalT > timeThreh:
                    coord = midCoord(i, j, mat)
                    stayPoint.append(
                        [i, j, coord[0], coord[1], mat.iloc[i, 0], mat.iloc[j, 0], mat.iloc[j, 0] - mat.iloc[i, 0]])
                    i = j
                    token = 1
                break
            j = j + 1
        if token != 1:
            i = i + 1
    SP = pd.DataFrame(stayPoint, columns=['i', 'j', 'x', 'y', 'arvT', 'levT', 'duration'])
    return SP  # dataframe格式


def Distance(i, j, mat):
    x0 = mat.iloc[i, 2]
    y0 = mat.iloc[i, 3]
    x1 = mat.iloc[j, 2]
    y1 = mat.iloc[j, 3]
    r = pow(pow(x1 - x0, 2) + pow(y1 - y0, 2), 0.5)
    return r


def midCoord(i, j, mat):
    mid = int((i + j) / 2)
    x = mat.iloc[i:j, 2].mean()
    y = mat.iloc[i:j, 3].mean()
    return x, y

def statistics_features(mat):
    timeThreh=300  #时间阈值
    distThreh=30   #距离阈值
    SP=detect_staypoints(timeThreh,distThreh,mat) #检测出的停顿点
    SPTime=0  #stay points的总停顿时间
    for i in range(len(SP)):
        SPTime=SPTime+(SP['levT'][i]-SP['arvT'][i])
    SPNum=len(SP) #stay points个数
    return SPNum,SPTime  #停顿次数、总停顿时间

def preprocess(data):
    datanew = pd.DataFrame(columns=data.columns)
    datanew=datanew.append(data.iloc[0,:])
    length=0
    for i in range(1,len(data)):
        if(data.iloc[i,0]==datanew.iloc[length,0] or (data.iloc[i,2]==datanew.iloc[length,2] and data.iloc[i,3]==datanew.iloc[length,3])):
            # print(data.iloc[i,2],data.iloc[i,3])
            # print(datanew.iloc[length,2],datanew.iloc[length,3])
            continue
        else:
            length=length+1
            datanew = datanew.append(data.iloc[i, :])
    return datanew


def classifier(dataset):
    '''
    第一种分类方法：对传进来的整段data直接使用分类器分类
    '''
    if (len(dataset) < 20 or dataset.iloc[0, 1] == -1):
        return 1
    else:
        dataset = preprocess(dataset)
        featureMat = np.zeros((1, 29))
        featureMat[0, 0] = dataset.iloc[-1, 0] - dataset.iloc[0, 0]  # 总时长
        featureDist = cal_featureDist(dataset)
        featureMat[0, 1] = featureDist[0]  # 总距离
        featureMat[0, 2] = featureDist[1]  # 最大距离
        featureMat[0, 3] = featureDist[2]  # 平均距离
        featureMat[0, 4] = featureDist[3]  # 距离标准差
        featureMat[0, 5] = featureDist[4]  # X轴总距离
        featureMat[0, 6] = featureDist[5]  # X轴最大距离
        featureMat[0, 7] = featureDist[6]  # X轴平均距离
        featureMat[0, 8] = featureDist[7]  # X轴距离标准差
        featureMat[0, 9] = featureDist[8]  # Y轴总距离
        featureMat[0, 10] = featureDist[9]  # Y轴最大距离
        featureMat[0, 11] = featureDist[10]  # Y轴平均距离
        featureMat[0, 12] = featureDist[11]  # Y轴距离标准差
        featureMat[0, 13] = x_Offset(dataset)  # X轴偏移量
        featureMat[0, 14] = y_Offset(dataset)  # Y轴偏移量
        featureSpeed = cal_featureSpeed(dataset)
        featureMat[0, 15] = featureSpeed[0]  # 最小速度
        featureMat[0, 16] = featureSpeed[1]  # 最大速度
        featureMat[0, 17] = featureSpeed[2]  # 平均速度
        featureMat[0, 18] = featureSpeed[3]  # 速度标准差
        featureMat[0, 19] = featureSpeed[4]  # X轴最大速度
        featureMat[0, 20] = featureSpeed[5]  # X轴平均速度
        featureMat[0, 21] = featureSpeed[6]  # X轴速度标准差
        featureMat[0, 22] = featureSpeed[7]  # Y轴最大速度
        featureMat[0, 23] = featureSpeed[8]  # Y轴平均速度
        featureMat[0, 24] = featureSpeed[9]  # Y轴速度标准差
        featureMat[0, 25] = avg_acc(dataset)  # 平均加速度
        featureMat[0, 26] = std_acc(dataset)  # 加速度标准差
        featureMat[0, 27] = avg_angle(dataset)  # 平均转角
        featureMat[0, 28] = featureDist[12]  # 滑动路径平滑度
        if (dataset.iloc[0, 1] == 14 or dataset.iloc[0, 1] == 15 or dataset.iloc[0, 1] == 17):  # 图像区域
            #columns=['TotalDist', 'MaxDist', 'AvgAngleChange', 'TotalTime', 'AvgSpeedY', 'AvgDist', 'MaxSpeedX', 'AvgAcc', 'SdDist']
            #columns=[1,2,27,0,9,3,19,25,4]
            #columns='AvgDist', 'TotalDist', 'MaxDist', 'MaxSpeed', 'AvgDistY', 'SdAcc', 'TotalDistY', 'MaxSpeedX', 'OffsetY', 'TotalDistX', 'AvgAcc', 'AvgSpeedY', 'MaxDistY', 'OffsetX', 'AvgDistX'
            columns=[3,1,2,16,11,26,9,19,14,5,25,23,10,13,7]
            feature_selected=[]
            for i in columns:
                feature_selected.append(featureMat[0,i])
            feature_selected=np.array(feature_selected).reshape(1,-1)
            result = browseImage.predict(feature_selected)
        elif (dataset.iloc[0, 1] == 13):  # 诊断报告区域
            #  ['AvgSpeed', 'TotalTime', 'SdSpeed', 'OffsetY', 'TotalDistY', 'AvgAngleChange',
            # 'TotalDistX', 'SdSpeedY', 'Str', 'TotalDist', 'SdSpeedX', 'SdDistY', 'SdDist',
            # 'SdDistX', 'SdAcc', 'OffsetX', 'MinSpeed', 'MaxSpeedY', 'MaxSpeedX', 'MaxDistX',
            # 'MaxDistY', 'MaxDist', 'AvgSpeedX', 'MaxSpeed', 'AvgSpeedY']
            columns=[17,0,18,14,9,27,5,24,28,1,21,12,4,8,26,13,15,22,19,6,10,2,20,16,11]
            feature_selected = []
            for i in columns:
                feature_selected.append(featureMat[0, i])
            feature_selected = np.array(feature_selected).reshape(1, -1)
            result = browseReport.predict(feature_selected)
        elif (dataset.iloc[0, 1] == 7): #工具箱
            # columns=['OffsetX', 'TotalTime', 'MaxDistX', 'AvgAcc', 'MinSpeed', 'OffsetY', 'AvgDistY', 'MaxSpeedY',
            #  'AvgAngleChange', 'MaxDist', 'MaxSpeed', 'TotalDistY', 'TotalDist', 'SdDistY', 'TotalDistX', 'SdSpeed']
            #columns=[13,0,6,25,15,14,11,22,27,2,16,9,1,12,5,18]
            #columns='OffsetX', 'AvgSpeed', 'TotalTime', 'TotalDist', 'AvgSpeedX', 'AvgAngleChange', 'MinSpeed', 'AvgAcc'
            columns=[13,17,0,1,20,27,15,25]
            feature_selected = []
            for i in columns:
                feature_selected.append(featureMat[0, i])
            feature_selected = np.array(feature_selected).reshape(1, -1)
            result = browseToolBox.predict(feature_selected)
        elif (dataset.iloc[0, 1] == 12):
            # ['AvgAngleChange', 'AvgSpeedY', 'MaxDistX', 'OffsetY', 'MaxDist', 'MaxDistY', 'Str', 'TotalDistY', 'SdDistY']
            #columns=[27,23,6,14,2,10,28,9,12]
            #columns='MaxDist', 'SdAcc', 'TotalDist', 'Str', 'SdDist', 'MaxDistY', 'AvgDistX', 'TotalDistY', 'TotalTime', 'AvgSpeedY', 'MaxDistX'
            columns=[2,26,1,28,4,10,7,9,0,23,6]
            feature_selected = []
            for i in columns:
                feature_selected.append(featureMat[0, i])
            feature_selected = np.array(feature_selected).reshape(1, -1)
            result = browseResultList.predict(feature_selected)
        else:
            return 1  #1表示背景事件
        return result

def classifier1(data):
    '''
    第二种分类方式：对整段data进行滑动窗口分段，对每段进行分类，再使用投票法进行分段结果表决
    '''
    if(len(data)<20 or data.iloc[0,1]==-1):
        return 0
    else:
        data = preprocess(data)
        num = int((len(data) - 120) / 30) + 1
        dataset = []
        if num <= 0:
            dataset.append(data)
            num = 1
        else:
            for i in range(num):
                data_n = data.iloc[i * 30:i * 30 + 120, :]
                dataset.append(data_n)
        featureMat = np.zeros((num, 29))
        for i in range(len(featureMat)):
            featureMat[i, 0] = dataset[i].iloc[-1, 0] - dataset[i].iloc[0, 0]  # 总时长
            featureDist = cal_featureDist(dataset[i])
            featureMat[i, 1] = featureDist[0]  # 总距离
            featureMat[i, 2] = featureDist[1]  # 最大距离
            featureMat[i, 3] = featureDist[2]  # 平均距离
            featureMat[i, 4] = featureDist[3]  # 距离标准差
            featureMat[i, 5] = featureDist[4]  # X轴总距离
            featureMat[i, 6] = featureDist[5]  # X轴最大距离
            featureMat[i, 7] = featureDist[6]  # X轴平均距离
            featureMat[i, 8] = featureDist[7]  # X轴距离标准差
            featureMat[i, 9] = featureDist[8]  # Y轴总距离
            featureMat[i, 10] = featureDist[9]  # Y轴最大距离
            featureMat[i, 11] = featureDist[10]  # Y轴平均距离
            featureMat[i, 12] = featureDist[11]  # Y轴距离标准差
            featureMat[i, 13] = x_Offset(dataset[i])  # X轴偏移量
            featureMat[i, 14] = y_Offset(dataset[i])  # Y轴偏移量
            featureSpeed = cal_featureSpeed(dataset[i])
            featureMat[i, 15] = featureSpeed[0]  # 最小速度
            featureMat[i, 16] = featureSpeed[1]  # 最大速度
            featureMat[i, 17] = featureSpeed[2]  # 平均速度
            featureMat[i, 18] = featureSpeed[3]  # 速度标准差
            featureMat[i, 19] = featureSpeed[4]  # X轴最大速度
            featureMat[i, 20] = featureSpeed[5]  # X轴平均速度
            featureMat[i, 21] = featureSpeed[6]  # X轴速度标准差
            featureMat[i, 22] = featureSpeed[7]  # Y轴最大速度
            featureMat[i, 23] = featureSpeed[8]  # Y轴平均速度
            featureMat[i, 24] = featureSpeed[9]  # Y轴速度标准差
            featureMat[i, 25] = avg_acc(dataset[i])  # 平均加速度
            featureMat[i, 26] = std_acc(dataset[i])  # 加速度标准差
            featureMat[i, 27] = avg_angle(dataset[i])  # 平均转角
            featureMat[i, 28] = featureDist[12]  # 滑动路径平滑度
        if (dataset[i].iloc[0, 1] == 14 or dataset[i].iloc[0, 1] == 15 or dataset[i].iloc[0, 1] == 17):  # 图像区域
            #columns=['TotalDist', 'MaxDist', 'AvgAngleChange', 'TotalTime', 'AvgSpeedY', 'AvgDist', 'MaxSpeedX', 'AvgAcc', 'SdDist']
            columns=[1,2,27,0,9,3,19,25,4]
            #columns = [3, 1, 2, 16, 11, 26, 9, 19, 14, 5, 25, 23, 10, 13, 7]
            feature_selected = []
            if num==1:
                for i in columns:
                    feature_selected.append(featureMat[0, i])
                feature_selected = np.array(feature_selected).reshape(1, -1)
            else:
                for i in columns:
                    feature_selected.append(featureMat[:,i])
                feature_selected=np.array(feature_selected).T
            result = browseImage.predict(feature_selected)
        elif (dataset[i].iloc[0, 1] == 13):  # 诊断报告区域
            #  ['AvgSpeed', 'TotalTime', 'SdSpeed', 'OffsetY', 'TotalDistY', 'AvgAngleChange',
            # 'TotalDistX', 'SdSpeedY', 'Str', 'TotalDist', 'SdSpeedX', 'SdDistY', 'SdDist',
            # 'SdDistX', 'SdAcc', 'OffsetX', 'MinSpeed', 'MaxSpeedY', 'MaxSpeedX', 'MaxDistX',
            # 'MaxDistY', 'MaxDist', 'AvgSpeedX', 'MaxSpeed', 'AvgSpeedY']
            columns=[17,0,18,14,9,27,5,24,28,1,21,12,4,8,26,13,15,22,19,6,10,2,20,16,11]
            feature_selected = []
            if num==1:
                for i in columns:
                    feature_selected.append(featureMat[0, i])
                feature_selected = np.array(feature_selected).reshape(1, -1)
            else:
                for i in columns:
                    feature_selected.append(featureMat[:,i])
                feature_selected=np.array(feature_selected).T
            result = browseReport.predict(feature_selected)
        elif (dataset[i].iloc[0, 1] == 7):  #工具箱
            # columns=['OffsetX', 'TotalTime', 'MaxDistX', 'AvgAcc', 'MinSpeed', 'OffsetY', 'AvgDistY', 'MaxSpeedY',
            #  'AvgAngleChange', 'MaxDist', 'MaxSpeed', 'TotalDistY', 'TotalDist', 'SdDistY', 'TotalDistX', 'SdSpeed']
            #columns=[13,0,6,25,15,14,11,22,27,2,16,9,1,12,5,18]
            #columns='OffsetX', 'AvgSpeed', 'TotalTime', 'TotalDist', 'AvgSpeedX', 'AvgAngleChange', 'MinSpeed', 'AvgAcc'
            columns = [13, 17, 0, 1, 20, 27, 15, 25]
            feature_selected = []
            if num==1:
                for i in columns:
                    feature_selected.append(featureMat[0, i])
                feature_selected = np.array(feature_selected).reshape(1, -1)
            else:
                for i in columns:
                    feature_selected.append(featureMat[:,i])
                feature_selected=np.array(feature_selected).T
            result = browseToolBox.predict(feature_selected)
        elif (dataset[i].iloc[0, 1] == 12): #ResultList
            # ['AvgAngleChange', 'AvgSpeedY', 'MaxDistX', 'OffsetY', 'MaxDist', 'MaxDistY', 'Str', 'TotalDistY', 'SdDistY']
            columns=[27,23,6,14,2,10,28,9,12]
            #columns = [2, 26, 1, 28, 4, 10, 7, 9, 0, 23, 6]
            feature_selected = []
            if num==1:
                for i in columns:
                    feature_selected.append(featureMat[0, i])
                feature_selected = np.array(feature_selected).reshape(1, -1)
            else:
                for i in columns:
                    feature_selected.append(featureMat[:,i])
                feature_selected=np.array(feature_selected).T
            result = browseResultList.predict(feature_selected)
        else:
            return 1  #1表示背景事件
        if (Counter(result)[0] > Counter(result)[1]):
            return 0
        else:
            return 1

