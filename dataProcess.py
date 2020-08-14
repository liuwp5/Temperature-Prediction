import pandas as pd

data_set = pd.read_csv("train/train.csv")
# 1、删去空白行
data_set.dropna(inplace=True)
# 2、截去时间信息
data_set = data_set.iloc[:, 7:13]
# 3、修正索引
data_set.reset_index(drop=True, inplace=True)

# data_set.to_csv('deleteSpace.csv')
# 4、去除偏差值较大的样本/误差样本
index = 1
# for index in range(1, df.shape[0]):   # range(1, 24389)
# toy = data_set.iloc[0:5700, :]
# toy.to_csv('ts.csv')
count = 0
# print(data_set.shape[0])  # 24389
while index < data_set.shape[0]:
# while index < toy.shape[0]:
    # print(count ,index)
    count += 1
    sub = abs(data_set.iloc[index] - data_set.iloc[index-1])

    for item in sub:
        if item > 30:
            # print(count, sub)
            data_set = data_set.drop(labels=index)

            index -= 1
            break
    index += 1
    data_set.reset_index(drop=True, inplace=True)

data_set.to_csv('correctData.csv')
# print(data_set.shape[0])   # 19203
# --------------------------------------------------------
# 5、合并
# start = 0
# end = 0
# result = pd.DataFrame(columns=['温度(室外)','湿度(室外)','气压(室外)','湿度(室内)','气压(室内)','temperature'])
# # for index in range(1, 10):
# for index in range(1, data_set.shape[0]):
#     if data_set.iloc[index][5] == data_set.iloc[index-1][5]:
#         end = index + 1
#     else:
#         len = end - start
#         # 合并
#         tmp = data_set[start:end][:].mean()
#         result = result.append(tmp, ignore_index=True)
#         start = index
#         end = index
#
# len = end - start
# # 合并最后一项
# tmp = data_set[start:end][:].mean()
# result = result.append(tmp, ignore_index=True)
#
# # 6、保留小数点后2位
# result = result.round(decimals=2)
#
# # 7、输出csv
# # result.to_csv('data.csv')
# # print(result.iloc[0])
# print(data_set.shape[0])