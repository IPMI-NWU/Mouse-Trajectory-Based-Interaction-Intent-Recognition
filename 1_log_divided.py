'''
由于原log包含一个用户对多个image的阅片过程数据，将其进行分割，使一个log对应一个image
'''
#-*- coding : utf-8-*-
# coding:unicode_escape
import pandas as pd
import os

def initial_data():  #读取log数据
    path = '../log_file/rawData/gt/'
    #path='../log_file/rawData/no_gt/'
    # 获取read_path下的所有文件名称（顺序读取的）
    files = os.listdir(path)
    for file_name in files:
        address = path + file_name
        name = os.path.splitext(file_name)[0]
        initial_data_1(address,name)


def initial_data_1(address,name):
    f = open(address, encoding='gbk')
    line = f.readline().strip()  # 读取第一行
    txt = []
    data = {}
    txt.append(line)
    while line:  # 直到读取完文件
        line = f.readline().strip()  # 读取一行文件，包括换行符
        txt.append(line)
    f.close()  # 关闭文件
    for i in range(len(txt) - 1):
        data[i] = txt[i].split()
    segmentation(data, name)

def segmentation(data,name):
    seg_data=[]
    trajectory=[]
    for i in range(len(data)):
        if(len(data[i])>3):
            if (data[i][3] == '7'):  # =7表示该条数据为一个切换图像动作，以此进行划分
                seg_data.append(trajectory)
                trajectory = []
        trajectory.append(data[i])
    seg_data.append(trajectory)
    for j in range(len(seg_data)):
        image_name=seg_data[j][-1][9]
        a = pd.DataFrame(seg_data[j])
        a=a.iloc[:,0:9]
        print(a.iloc[0])
        a.to_csv('../log_file/seg_rawdata/gt/%s.csv' %name, index=False,encoding="utf_8_sig",header=None)
        #a.to_csv('../log_file/seg_rawdata/no_gt/%s_%s.csv' % (name, image_name), index=False)


if __name__ == '__main__':
    initial_data()


