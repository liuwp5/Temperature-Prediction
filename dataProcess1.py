import pandas as pd
import random


def ClearDirtyData(data_set):
    # 1、删去空白行
    data_set.dropna(inplace=True)

    # 2、修正索引
    data_set.reset_index(drop=True, inplace=True)
    # 3、去除偏差值较大的样本/误差样本
    index = 1
    # count = 0
    # print(data_set.shape[0])  # 24389
    while index < data_set.shape[0]:
        # print(count ,index)
        sub = abs(data_set.iloc[index][7:13] - data_set.iloc[index-1][7:13])

        for item in sub:
            if item > 30:
                # print(count, sub)
                data_set = data_set.drop(labels=index)

                index -= 1
                break
        index += 1
        data_set.reset_index(drop=True, inplace=True)
    # 4、保存去除脏数据后
    data_set.to_csv('correctData1.csv')
    # print(data_set.shape[0])   # 19203
    return data_set


def SampleData(data_set, sampledata):
    # [4][5][6]hms

    start, end = 0, 0
    half = 0
    index = 0
    while index < data_set.shape[0]:
        hour_now = data_set.iloc[index][4]
        minute_now = data_set.iloc[index][5]

        if half == 0 and minute_now < 30:
            end = index
        elif half == 0 and minute_now >= 30:
            # Add(start, end, sampledata, data_set)

            random_index = random.randint(start, end)

            sample = data_set.iloc[random_index]
            print(sample)
            sampledata = sampledata.append(sample, ignore_index=True)

            start = index
            end = index
            half = 1
        elif half == 1 and minute_now >= 30:
            end = index
        elif half == 1 and minute_now < 30:
            # Add(start, end, sampledata, data_set)

            random_index = random.randint(start, end)
            sample = data_set.iloc[random_index]
            sampledata = sampledata.append(sample, ignore_index=True)

            start = index
            end = index
            half = 0
        index += 1
        # print(index)
    # Add(start, end, sampledata, data_set)

    random_index = random.randint(start, end)
    sample = data_set.iloc[random_index]
    sampledata = sampledata.append(sample, ignore_index=True)

    sampledata.to_csv('sampleData.csv')
    print(sampledata.shape)


if __name__ == '__main__':
    # hyper

    # 1、读取训练集
    data_set = pd.read_csv("train/train.csv")

    # 2、清洗数据
    data_set = ClearDirtyData(data_set) # 23503

    # 3、半小时采样一次
    sampledata = pd.DataFrame(
        columns=['time', '年', '月', '日', '小时', '分钟', '秒', '温度(室外)', '湿度(室外)', '气压(室外)', '湿度(室内)', '气压(室内)',
                 'temperature'])
    SampleData(data_set, sampledata)