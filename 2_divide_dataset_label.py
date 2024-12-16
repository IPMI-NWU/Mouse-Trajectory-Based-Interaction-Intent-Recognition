'''
对一条数字阅片数据进行处理,将raw_data划分为数据集+label的形式，抽取其中的标签
'''

# 注意read_csv和to_csv操作时的header问题，可能会默认第一行为列名
import pandas as pd
import time
import json
import os


def initial_data():  #读取log数据
    path = '../log_file/seg_rawdata/gt/'
    #path = '../log_file/seg_rawdata/no_gt/'
    # 获取read_path下的所有文件名称（顺序读取的）
    files = os.listdir(path)
    for file_name in files:
        address = path + file_name
        name = os.path.splitext(file_name)[0]
        data = pd.read_csv(address)
        process(data, name)


def timestamp(date,clock):
    second = time.mktime(time.strptime(date + ' ' + clock, "%Y-%m-%d %H:%M:%S,%f"))
    millisecond = int(clock.split(',')[1])  # 截取毫秒部分
    stamp=second*1000+millisecond
    return stamp

def process(data,filename):
    new_data={}
    annotions=[]
    j=0
    delete_num=0
    els = list(data.items())
    start_time= time.mktime(time.strptime(data.iloc[0,0]+' '+data.iloc[0,1], "%Y-%m-%d %H:%M:%S,%f"))
    start_time_1=int(data.iloc[0,1].split(',')[1])  #截取毫秒部分
    end_time = time.mktime(time.strptime(data.iloc[-1,0] + ' ' + data.iloc[-1,1], "%Y-%m-%d %H:%M:%S,%f"))
    end_time_1 = int(data.iloc[-1,1].split(',')[1])  # 截取毫秒部分
    #整条轨迹开始和结束的时间
    duration=(end_time*1000+end_time_1)-(start_time*1000+start_time_1)
    for i in range(len(data)):
        if (data.iloc[i,2] == 'START'):
            annotions_start_time=timestamp(data.iloc[i,0],data.iloc[i,1])
            annotions_start_time=annotions_start_time-(start_time*1000+start_time_1)
            delete_num+=1
            start_index=i-delete_num+1  #'START'的下一条
        elif(data.iloc[i,2]=='END'):
            annotions_end_time=timestamp(data.iloc[i,0],data.iloc[i,1])
            annotions_end_time=annotions_end_time-(start_time*1000+start_time_1)
            end_index=i-delete_num-1 #'END'的上一条
            delete_num += 1
            annotions.append({'label':data.iloc[i,3],'segment':[annotions_start_time,annotions_end_time,start_index,end_index,(end_index-start_index+1)]})

        else:
            new_data[j]=data.loc[i]  #new_data是删除groundtruth_label后的数据
            j=j+1
    label={filename:{'duration':duration,'num_reading':len(new_data),'num_label':len(annotions),'information':annotions}}
    info_json = json.dumps(label, sort_keys=False, indent=4, separators=(',', ': '),ensure_ascii=False)
    f = open('label/' + '%s.json' % filename, 'a+', encoding='utf-8')
    f.write(info_json)
    new_data = pd.DataFrame.from_dict(new_data, orient='index')
    new_data.to_csv('0_dataset/'+'%s.csv'%filename,index=False)


if __name__ == '__main__':
    initial_data()

