#
# 在基础事件上进一步划分高级语义事件
# ['7', '浏览辅助诊断工具箱', '1377.0', '2209.0', '-1', '0', '0', '512', '512', '0', '512', '0', '512', '512', '512', '0', '512', '0', '512']
# [粗粒度事件编号,事件名称，开始时间，结束时间，event_object,image_view,result_view...]
import pandas as pd
import numpy as np
import res_tool
import os

semantic_event=res_tool.semantic_event
df_box = pd.read_csv('../utils/bounding_box.csv',encoding='utf-8')
image_name=''

class prediction_behavior:
    def __init__(self, Coarse_grained_type,Coarse_grained_name,
                 start_time,end_time,object,view_para,result_para,
                 start_image_size,start_image_view,
                 end_image_size,end_image_view,fine_grained_type=-1,
                 fine_grained_name=None,medical_region=0,aoi_region=0):
        self.Coarse_grained_type=Coarse_grained_type
        self.Coarse_grained_name=Coarse_grained_name
        self.start_time=start_time
        self.end_time=end_time
        self.object=object
        self.view_para=view_para
        self.result_para=result_para
        self.start_image_size=start_image_size  #[512,512]
        self.start_image_view=start_image_view  #[0,512,0,512]
        self.end_image_size=end_image_size
        self.end_image_view = end_image_view
        self.fine_grained_type=fine_grained_type
        self.fine_grained_name=fine_grained_name
        self.medical_region=medical_region
        self.aoi_region=aoi_region

def initial_data(df):  #经过处理的prediction行为列表
    df.iloc[-1,-1]=512
    prediction=[]
    for i in range(len(df)):
        prediction.append(prediction_behavior(int(df.iloc[i,0]),df.iloc[i,1],float(df.iloc[i,2]),float(df.iloc[i,3]),int(df.iloc[i,4]),int(df.iloc[i,5]),int(df.iloc[i,6]),
                          [int(df.iloc[i,7]),int(df.iloc[i,8])],[int(df.iloc[i,9]),int(df.iloc[i,10]),int(df.iloc[i,11]),int(df.iloc[i,12])],
                          [int(df.iloc[i,13]),int(df.iloc[i,14])],[int(df.iloc[i,15]),int(df.iloc[i,16]),int(df.iloc[i,17]),int(df.iloc[i,18])]))#对每个预测行为实例化
    return prediction


def refine_event(data):  #根据已经划分事件类型,考虑一些额外的属性，划分详细的高级语义事件
    image=0  #默认为原始图像
    for i in range(len(data)):
        type=data[i].Coarse_grained_type
        if(type in [1,2,3,4,5,6,7]):
            data[i].fine_grained_type=type
            if type==7:
                data[i].aoi_region==7
        elif(type==8):   #基础事件：使用XXX工具检测
            data[i].aoi_region=8
            object=data[i].object
            if object in (2,3,4,13):
                data[i].medical_region = 1  #胸廓
            elif object in (5,15):
                data[i].medical_region = 2 #气管
            elif object in (1,10,11,12):
                data[i].medical_region = 3 #肺野
            elif object in (6,9,14):
                data[i].medical_region = 4 #纵隔肺门
            elif object in (7,8):
                data[i].medical_region = 5  #横隔
            elif object==0:
                data[i].medical_region=6 #肺结核检测

            if(object==0):  #操作对象属性==0，表示使用肺结核检测工具
                data[i].fine_grained_type=8
            elif(object==1):
                data[i].fine_grained_type = 9
            elif (object == 2):
                data[i].fine_grained_type = 10
            elif (object == 3):
                data[i].fine_grained_type = 11
            elif (object == 4):
                data[i].fine_grained_type = 12
            elif (object == 5):
                data[i].fine_grained_type = 13
            elif (object == 6):
                data[i].fine_grained_type = 14
            elif (object == 7):
                data[i].fine_grained_type = 15
            elif (object == 8):
                data[i].fine_grained_type = 16
            elif (object == 9):
                data[i].fine_grained_type = 17
            elif (object == 10):
                data[i].fine_grained_type = 18
            elif (object == 11):
                data[i].fine_grained_type = 19
            elif (object == 12):
                data[i].fine_grained_type = 20
            elif (object == 13):
                data[i].fine_grained_type = 21
            elif (object == 14):
                data[i].fine_grained_type = 22
            elif (object == 15):
                data[i].fine_grained_type = 23
        elif (type == 9):
            data[i].fine_grained_type = 24
        elif (type == 10):
            object = data[i].object
            data[i].aoi_region=7
            if(object==0):
                data[i].fine_grained_type = 25
                data[i].medical_region=6 #肺结核检测相关
            elif(object==1):
                data[i].fine_grained_type = 26
                data[i].medical_region=1 #胸廓相关
            elif (object == 2):
                data[i].fine_grained_type = 27
                data[i].medical_region=3 #肺野相关
            elif (object == 3):
                data[i].fine_grained_type = 28
                data[i].medical_region=1
            elif (object == 4):
                data[i].fine_grained_type = 29
                data[i].medical_region=2 #气管相关
        elif (type == 11):   #浏览XXX诊断报告
            result_para=data[i].result_para
            data[i].aoi_region=13
            if result_para in (2,3,4,13):
                data[i].medical_region = 1  #胸廓
            elif result_para in (5,15):
                data[i].medical_region = 2 #气管
            elif result_para in (1,10,11,12):
                data[i].medical_region = 3 #肺野
            elif result_para in (6,9,14):
                data[i].medical_region = 4 #纵隔肺门
            elif result_para in (7,8):
                data[i].medical_region = 5  #横隔

            if (result_para == 0):  # result属性==0，表示为肺结核检测结果
                data[i].fine_grained_type = 30
            elif (result_para == 6):
                data[i].fine_grained_type = 31
            elif (result_para == 7):
                data[i].fine_grained_type = 32
            elif (result_para == 8):
                data[i].fine_grained_type = 33
            elif (result_para == 9):
                data[i].fine_grained_type = 34
            elif (result_para == 10):
                data[i].fine_grained_type = 35
            elif (result_para == 11):
                data[i].fine_grained_type = 36
            elif (result_para == 12):
                data[i].fine_grained_type = 37
            elif (result_para == 13):
                data[i].fine_grained_type = 38
            elif (result_para == 14):
                data[i].fine_grained_type = 39
            elif (result_para == 15):
                data[i].fine_grained_type = 40
        elif (type == 12):  #浏览检测结果列表
            data[i].fine_grained_type=41
            data[i].aoi_region=12
        elif (type==13):   #浏览XXX检测结果图像
            object=data[i].object
            data[i].aoi_region = 12
            if object in (2,3,4,13):
                data[i].medical_region = 1  #胸廓
            elif object in (5,15):
                data[i].medical_region = 2 #气管
            elif object in (1,10,11,12):
                data[i].medical_region = 3 #肺野
            elif object in (6,9,14):
                data[i].medical_region = 4 #纵隔肺门
            elif object in (7,8):
                data[i].medical_region = 5  #横隔

            if (object == 0):  # 操作对象属性==0，表示使用肺结核检测工具
                data[i].fine_grained_type = 42
            elif (object == 1):
                data[i].fine_grained_type = 43
            elif (object == 2):
                data[i].fine_grained_type = 44
            elif (object == 3):
                data[i].fine_grained_type = 45
            elif (object == 4):
                data[i].fine_grained_type = 46
            elif (object == 5):
                data[i].fine_grained_type = 47
            elif (object == 6):
                data[i].fine_grained_type = 48
            elif (object == 7):
                data[i].fine_grained_type = 49
            elif (object == 8):
                data[i].fine_grained_type = 50
            elif (object == 9):
                data[i].fine_grained_type = 51
            elif (object == 10):
                data[i].fine_grained_type = 52
            elif (object == 11):
                data[i].fine_grained_type = 53
            elif (object == 12):
                data[i].fine_grained_type = 54
            elif (object == 13):
                data[i].fine_grained_type = 55
            elif (object == 14):
                data[i].fine_grained_type = 56
            elif (object == 15):
                data[i].fine_grained_type = 57
        elif (type == 14):  # 浏览诊断图像
            data[i].fine_grained_type = 58
            data[i].aoi_region = 14
        elif(type==15):  #浏览辅助检测图像
            result_para = data[i].result_para
            data[i].aoi_region = 15
            if result_para in (2,3,4,13):
                data[i].medical_region = 1  #胸廓
            elif result_para in (5,15):
                data[i].medical_region = 2 #气管
            elif result_para in (1,10,11,12):
                data[i].medical_region = 3 #肺野
            elif result_para in (6,9,14):
                data[i].medical_region = 4 #纵隔肺门
            elif result_para in (7,8):
                data[i].medical_region = 5  #横隔

            if (result_para == 0):
                data[i].fine_grained_type = 59
            elif (result_para == 1):
                data[i].fine_grained_type = 60
            elif (result_para == 2):
                data[i].fine_grained_type = 61
            elif (result_para == 3):
                data[i].fine_grained_type = 62
            elif (result_para == 4):
                data[i].fine_grained_type = 63
            elif (result_para == 5):
                data[i].fine_grained_type = 64
            elif (result_para == 6):
                data[i].fine_grained_type = 65
            elif (result_para == 7):
                data[i].fine_grained_type = 66
            elif (result_para == 8):
                data[i].fine_grained_type = 67
            elif (result_para == 9):
                data[i].fine_grained_type = 68
            elif (result_para == 10):
                data[i].fine_grained_type = 69
            elif (result_para == 11):
                data[i].fine_grained_type = 70
            elif (result_para == 12):
                data[i].fine_grained_type = 71
            elif (result_para == 13):
                data[i].fine_grained_type = 72
            elif (result_para == 14):
                data[i].fine_grained_type = 73
            elif (result_para == 15):
                data[i].fine_grained_type = 74
        elif (type == 16):  # 进入图像详情
            data[i].fine_grained_type = 75
            if(data[i].object==1):
                image=1 #表示打开“检测结果图像”详情
                data[i].aoi_region=15
            else:
                data[i].aoi_regon=16
        elif (type == 17):  # 退出图像详情
            data[i].fine_grained_type = 76
            data[i].aoi_region=17
#---------------------------放大+缩小-----------------------------------
        elif(type in [18,19]):
            data[i].aoi_region = 17
            number = (type - 18) * 16
            if(image==0):
                data[i].fine_grained_type = number + 77
            else:
                result_para = data[i].result_para
                if result_para in (2, 3, 4, 13):
                    data[i].medical_region = 1  # 胸廓
                elif result_para in (5, 15):
                    data[i].medical_region = 2  # 气管
                elif result_para in (1, 10, 11, 12):
                    data[i].medical_region = 3  # 肺野
                elif result_para in (6, 9, 14):
                    data[i].medical_region = 4  # 纵隔肺门
                elif result_para in (7, 8):
                    data[i].medical_region = 5  # 横隔

                if (result_para == 0):
                    data[i].fine_grained_type = number + 78
                elif (result_para == 1):
                    data[i].fine_grained_type = number + 79
                elif(result_para in [6,7,8,9,10,11,12]):
                    data[i].fine_grained_type = number + 74+data[i].result_para
                elif (result_para == 14):
                    data[i].fine_grained_type = number + 87
                elif (result_para == 2):
                    data[i].fine_grained_type = number + 88
                elif (result_para == 3):
                    data[i].fine_grained_type = number + 89
                elif (result_para == 4):
                    data[i].fine_grained_type = number + 90
                elif (result_para == 13):
                    data[i].fine_grained_type = number + 91
                elif (result_para in [5,15]):
                    data[i].fine_grained_type = number + 92
# ---------------------------移动+浏览-----------------------------------
        elif(type in [20,21]):  #放大图像
            data[i].aoi_region = 17
            number=(type-20)*53
            if(image==0):  #对“原始图像”进行详情查看
                region=region_1(data[i].start_image_size,data[i].start_image_view,
                                data[i].end_image_size, data[i].end_image_view)
                data[i].fine_grained_type = number+109+region
            else:          #对“诊断图像”进行详情查看
                result_para = data[i].result_para
                if result_para in (2, 3, 4, 13):
                    data[i].medical_region = 1  # 胸廓
                elif result_para in (5, 15):
                    data[i].medical_region = 2  # 气管
                elif result_para in (1, 10, 11, 12):
                    data[i].medical_region = 3  # 肺野
                elif result_para in (6, 9, 14):
                    data[i].medical_region = 4  # 纵隔肺门
                elif result_para in (7, 8):
                    data[i].medical_region = 5  # 横隔

                if(result_para==0):
                    region = region_1(data[i].start_image_size, data[i].start_image_view,
                                      data[i].end_image_size, data[i].end_image_view)
                    data[i].fine_grained_type = number+115 + region
                elif(result_para==1):
                    region = region_2(data[i].start_image_size, data[i].start_image_view,
                                      data[i].end_image_size, data[i].end_image_view)
                    data[i].fine_grained_type = number+121 + region
                elif (result_para == 6):
                    region = region_2(data[i].start_image_size, data[i].start_image_view,
                                      data[i].end_image_size, data[i].end_image_view)
                    data[i].fine_grained_type = number+124 + region
                elif (result_para == 7):
                    region = region_2(data[i].start_image_size, data[i].start_image_view,
                                      data[i].end_image_size, data[i].end_image_view)
                    data[i].fine_grained_type = number+127+ region
                elif (result_para == 8):
                    region = region_2(data[i].start_image_size, data[i].start_image_view,
                                      data[i].end_image_size, data[i].end_image_view)
                    data[i].fine_grained_type = number+130 + region
                elif (result_para == 9):
                    region = region_2(data[i].start_image_size, data[i].start_image_view,
                                      data[i].end_image_size, data[i].end_image_view)
                    data[i].fine_grained_type = number+133 + region
                elif (result_para == 10):
                    region = region_2(data[i].start_image_size, data[i].start_image_view,
                                      data[i].end_image_size, data[i].end_image_view)
                    data[i].fine_grained_type = number+136 + region
                elif (result_para == 11):
                    region = region_2(data[i].start_image_size, data[i].start_image_view,
                                      data[i].end_image_size, data[i].end_image_view)
                    data[i].fine_grained_type =number+139 + region
                elif (result_para == 12):
                    region = region_2(data[i].start_image_size, data[i].start_image_view,
                                      data[i].end_image_size, data[i].end_image_view)
                    data[i].fine_grained_type = number+142 + region
                elif (result_para == 14):
                    region = region_2(data[i].start_image_size, data[i].start_image_view,
                                      data[i].end_image_size, data[i].end_image_view)
                    data[i].fine_grained_type = number+145 + region
                elif (result_para == 2):
                    region = region_2(data[i].start_image_size, data[i].start_image_view,
                                      data[i].end_image_size, data[i].end_image_view)
                    data[i].fine_grained_type = number+148 + region
                elif (result_para == 3):
                    region = region_2(data[i].start_image_size, data[i].start_image_view,
                                      data[i].end_image_size, data[i].end_image_view)
                    data[i].fine_grained_type = number+151 + region
                elif (result_para == 4):
                    region = region_2(data[i].start_image_size, data[i].start_image_view,
                                      data[i].end_image_size, data[i].end_image_view)
                    data[i].fine_grained_type = number+154 + region
                elif (result_para == 13):
                    region = region_2(data[i].start_image_size, data[i].start_image_view,
                                      data[i].end_image_size, data[i].end_image_view)
                    data[i].fine_grained_type = number+157 + region
                elif(result_para==5):#器官分割图
                    data[i].fine_grained_type = number+160
                elif (result_para == 14):
                    data[i].fine_grained_type = number+161
    return data

def iou(box1, box2):
    '''
    两个框（二维）的 iou 计算
    注意：边框以左上为原点
    box:[x1,y1,x2,y2],依次为左上右下坐标
    '''
    h = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    w = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area_box1 = ((box1[2] - box1[0]) * (box1[3] - box1[1]))
    area_box2 = ((box2[2] - box2[0]) * (box2[3] - box2[1]))
    inter = w * h
    union = area_box1 + area_box2 - inter
    iou = inter / union
    #print(inter, union, iou)
    return iou

def region_1(start_image_size,start_image_view,end_image_size,end_image_view):
    '''
     对原始诊断图像（完整版）判断操作区域
     return region(0：气管 1：左肺 2：右肺 3：上纵隔 4：心脏剪影/心脏轮廓 5：整体原图)
    '''
    view = [end_image_view[0], end_image_view[2], end_image_view[1], end_image_view[3]]  # 将image_view变为左上、右下的xy坐标
    w_h = end_image_size[0]
    # right lung
    rl_box = df_box[image_name][0:4].tolist()  # 原本从dataframe中读取的数据为series类型，将其转为list类型
    # left lung
    ll_box = df_box[image_name][4:8].tolist()
    # upper mediastinum
    um_box = df_box[image_name][8:12].tolist()
    # cardiac silhouette
    cs_box = df_box[image_name][12:16].tolist()
    # trachea
    tr_box = df_box[image_name][16:20].tolist()
    iou_rl = iou(view, rl_box)
    iou_ll = iou(view, ll_box)
    iou_um = iou(view, um_box)
    iou_cs = iou(view, cs_box)
    iou_tr = iou(view, tr_box)
    iou_overall = iou(view, [0, 0, w_h, w_h])
    max_iou = max(iou_rl, iou_ll, iou_um, iou_cs, iou_tr, iou_overall)
    if (max_iou == iou_tr):
        return 0
    elif (max_iou == iou_ll):
        return 1
    elif (max_iou == iou_rl):
        return 2
    elif (max_iou == iou_um):
        return 3
    elif (max_iou == iou_cs):
        return 4
    elif (max_iou == iou_overall):
        return 5

#
# 对分割图像判断操作区域
# return region(0：左侧 1：右侧 2：整体原图)
#
def region_2(start_image_size,start_image_view,end_image_size,end_image_view): #image_size:[512,512] image_view:[0,512,0,512]
    w_h = end_image_size[0]
    view = [end_image_view[0], end_image_view[2], end_image_view[1], end_image_view[3]]  # 将image_view变为左上、右下的xy坐标
    iou1 = iou(view, [0, 0, w_h, w_h]) #计算当前view与整体图像的iou
    iou2 = iou(view, [0, 0, w_h / 2, w_h]) #计算当前view与图像左半部分的iou
    iou3 = iou(view, [w_h / 2, 0, w_h, w_h]) #计算当前view与图像右半部分的iou
    max_iou = max(iou1, iou2, iou3)
    if (max_iou == iou1):
        return 2
    elif (max_iou == iou2):
        return 0
    else:
        return 1

def save(event,filename):
    data=[]
    for x in event:
        data.append([x.Coarse_grained_type,x.Coarse_grained_name,x.fine_grained_type,semantic_event[x.fine_grained_type],
                     x.start_time,x.end_time,x.medical_region,x.aoi_region])
    for i in data:
        print(i)
    df_new = pd.DataFrame(data,columns=['coarse_number', 'coarse_name', 'fine_number', 'fine_name',
                                        'start_time', 'end_time','medical_region','aoi_region'])
    path='../log_process/3_fine_event/'
    address=path+filename+'.csv'
    #print(df_new)
    df_new.to_csv(address, index=False, header=True,encoding='utf_8_sig')



if __name__ == '__main__':
    path='2_coarse_event/postprocessed/'
    files = os.listdir(path)
    for file_name in files:
        name = os.path.splitext(file_name)[0]
        image_name=name.split('_')[-1]
        df = pd.read_csv(path + name + '.csv')
        data = initial_data(df)  # 返回的data是一个na'me列表，列表中的每个元素是预测行为实例化对象
        data = refine_event(data)
        save(data, os.path.splitext(file_name)[0])


