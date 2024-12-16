import time
import pandas as pd
import os

def initial_data(address):  #读取log数据
    data = pd.read_csv(address)
    return data

def timestamp(data):#将原log数据中的日期转换为时间戳
    millisecond=[]
    for i in range(len(data)):
        str = int(data.iloc[i,1].split(',')[1])  #截取毫秒部分
        data.iloc[i,0] = data.iloc[i,0]+' '+data.iloc[i,1]  #然后换算成时间戳（最小单位为秒）
        data.iloc[i,0]= time.mktime(time.strptime(data.iloc[i,0], "%Y-%m-%d %H:%M:%S,%f")) #转时间戳
        data.iloc[i,1] = -1  #这一列用来表示该点划分的区域
        millisecond.append(str)
    start_time=data.iloc[0,0]; #秒的部分
    start_time_1=millisecond[0]; #毫秒的部分
    for i in range(len(data)):
        # if(i==0):
        #     start_time=data.iloc[0,0]*1000+millisecond[0]
        #     data.iloc[0,0]=0
        # else:
        data.iloc[i,0]=(data.iloc[i,0]-start_time)*1000+(millisecond[i]-start_time_1)
    return data

def divide_region(data):#给鼠标轨迹事件划分region
    for i in range(len(data)):
        s=data.iloc[i,2]  #读取坐标属性，此时类型为字符串
        if(s!='-1'):
            s = s[1:-1]  # 删除括号
            s = s.split(',')
            x = int(float(s[0]))  # 将字符串类型转换为整数，表示x,y坐标
            y = int(float(s[1]))
            if (y >= 25 and y <= 66):
                if (x >= 20 and x <= 115):
                    data.iloc[i,1] = 1
                elif (x > 115 and x <= 210):
                    data.iloc[i,1] = 2
                elif (x > 210 and x <= 305):
                    data.iloc[i,1] = 3
                elif (x > 305 and x <= 400):
                    data.iloc[i,1] = 4
                elif (x > 400 and x <= 495):
                    data.iloc[i,1] = 5
                elif (x > 495 and x <= 590):
                    data.iloc[i,1] = 6
            if (x >= 20 and x <= 290):
                if (y >= 90 and y <= 451):
                    data.iloc[i,1] = 7
                elif (y >= 470 and y <= 505):
                    if (x >= 180 and x <= 290):
                        data.iloc[i,1] = 8
                elif (y >= 528 and y <= 809):
                    data.iloc[i,4] = 9
                elif (y >= 825 and y <= 860):
                    if (x >= 20 and x <= 130):
                        data.iloc[i,1] = 10
                    if (x >= 180 and x <= 290):
                        data.iloc[i,1] = 11
            if (x >= 1442 and x <= 1752):
                if (y >= 90 and y <= 462):
                    data.iloc[i,1] = 12
                elif (y >= 486 and y <= 858):
                    data.iloc[i,1] = 13
            if ((x >= 310 and x <= 1421) and (y >= 70 and y <= 861)):
                data.iloc[i,1] = 16
                if (data.iloc[i,5] ==0):  # 全局视图
                    if ((x >= 340 and x <= 841) and (y >= 140 and y <= 741)):
                        data.iloc[i,1] = 14
                    if ((x >= 890 and x <= 1391) and (y >= 140 and y <= 741)):
                        data.iloc[i,1] = 15
                else:  # 图像详情视图
                    data.iloc[i,1] = 18
                    if ((x >= 481 and x <= 1249) and (y >= 82 and y <= 850)):
                        data.iloc[i,1] = 17
        else:  #只有当双击qlabel进入/退出图像详情模式时，记录的轨迹点为-1
            if(data.iloc[i,3]==7):  #点击图像列表进行切换
                data.iloc[i,1] = 9
            elif(data.iloc[i,3]==8):  #点击结果列表切换结果
                data.iloc[i,1] = 12
            elif(data.iloc[i,3]==2):  #双击
                if (data.iloc[i,4] == -1):  # 此时表示退出图像详情模型
                    data.iloc[i,1] = 17
                elif (data.iloc[i,4] == 0):  # 此时表示双击原始图像进行图像详情模式
                    data.iloc[i,1] = 14
                else:
                    data.iloc[i,1] = 15
    return data

def processing(data):  #对日志数据做预处理，转换数据类型
    newdata={}
    for i in range(len(data)):
        newdata[i]=[]
        newdata[i].append(data.iloc[i,0])  #时间戳
        newdata[i].append(data.iloc[i,1])
        s=data.iloc[i,2]
        if (s!= '-1'):
            s = s[1:-1]  # 删除'()'
            s = s.split(',')
            newdata[i].append(int(float(s[0])))  # 将字符串类型转换为整数，表示x,y坐标
            newdata[i].append(int(float(s[1])))
        else:
            newdata[i].append(-1)  # 将字符串类型转换为整数，表示x,y坐标
            newdata[i].append(-1)
        newdata[i].append(int(data.iloc[i,3])) #鼠标操作类型
        newdata[i].append(int(data.iloc[i,4])) #鼠标操作对象
        newdata[i].append(int(data.iloc[i,5])) #视图参数
        newdata[i].append(int(data.iloc[i,6])) #结果参数
        s=data.iloc[i,7]
        s=s.split('*')
        newdata[i].append(int(float(s[0])))  #当前图像的宽
        newdata[i].append(int(float(s[1])))  #当前图像的长
        s=data.iloc[i,8]
        if (s != '[]'):
            s = s[1:-1]  # 删除'[]'
            s = s.split(',')
            newdata[i].append(int(float(s[0])))    #图像详情模式时，当前显示图像为原图像的哪一区域
            newdata[i].append(int(float(s[1])))
            newdata[i].append(int(float(s[2])))
            newdata[i].append(int(float(s[3])))
        else:
            newdata[i].append(0)
            newdata[i].append(512)
            newdata[i].append(0)
            newdata[i].append(512)
        newdata[i].append(0)  #表示interval_time
    return newdata


if __name__ == '__main__':
    path = '0_dataset/without_label/'
    # path='../log_file/seg_rawdata/no_gt/'  #没有记录groundtruth的数据可以直接进行预处理
    # 获取read_path下的所有文件名称（顺序读取的）
    files = os.listdir(path)
    for file_name in files:
        print(file_name)
        address = path + file_name
        name = os.path.splitext(file_name)[0]
        dataset = initial_data(address)
        dataset = timestamp(dataset)
        dataset_region = divide_region(dataset)
        newdata = processing(dataset_region)
        dataset = pd.DataFrame.from_dict(newdata, orient='index',
                                         columns=['timestamp', 'region', 'x', 'y', 'mouse_type', 'mouse_object',
                                                  'view_parm', 'result_parm', 'image_width', 'image_height',
                                                  'x_start', 'x_end', 'y_start', 'y_end', 'interval_time',
                                                  ])
        for i in range(len(dataset)):
            if (i == 0):
                newdata[i].append(0)
            else:
                interval = dataset.iloc[i, 0] - dataset.iloc[i - 1, 0]
                dataset.iloc[i, 14] = interval
        dataset.to_csv('1_preprocess/without_label/'+'%s.csv'%name, index=False, header=True)

