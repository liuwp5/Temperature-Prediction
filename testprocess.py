import numpy as np


'''
train.csv and test.csv file
the header of these two files are:
time,年,月,日,小时,分钟,秒,温度(室外),湿度(室外),气压(室外),湿度(室内),气压(室内),temperature
totally 13 columns
'''
######## read test.csv
f = open('./test/test.csv', encoding='utf-8')
f_content = f.readlines()
f.close()

total_samples = len(f_content)-1

samples = []

for i, line in enumerate(f_content):
    if i >= 1:
        current_line_split = line.strip().split(',')
        current_line_split.append(0)
        samples.append(current_line_split[-6:])

samples = np.stack(samples)

######## read validate0(last 47 lines)
f1 = open('./train/validate0', encoding='utf-8')
f1_content = f1.readlines()
f1.close()

total_samples1 = len(f1_content)-1

samples1 = []

for i, line in enumerate(f1_content):
    if i > total_samples1-47:
        current_line_split = line.strip().split(',')
        samples1.append(current_line_split)

samples1 = np.stack(samples1)

# print('size:', samples1.shape)


############## calculate the average value of the features to get rid of the outliers
outdoor_tem = samples[:, 0]
outdoor_humidity = samples[:, 1]
outdoor_pressure = samples[:, 2]
indoor_humidity = samples[:, 3]
indoor_pressure = samples[:, 4]
indoor_tem = samples[:, 5]

ave_out_tem, ave_out_hum, ave_out_pre, ave_in_hum, ave_in_pre = [], [], [], [], []

for i in range(len(outdoor_tem)):
    if outdoor_tem[i] != '':
        ave_out_tem.append(float(outdoor_tem[i]))
    if outdoor_humidity[i] != '':
        ave_out_hum.append(float(outdoor_humidity[i]))
    if outdoor_pressure[i] != '':
        ave_out_pre.append(float(outdoor_pressure[i]))
    if indoor_humidity[i] != '':
        ave_in_hum.append(float(indoor_humidity[i]))
    if indoor_pressure[i] != '':
        ave_in_pre.append(float(indoor_pressure[i]))
    # if indoor_tem[i] != '':
    #     ave_in_tem.append(float(indoor_tem[i]))

ave = []
ave.append(np.mean(ave_out_tem))
ave.append(np.mean(ave_out_hum))
ave.append(np.mean(ave_out_pre))
ave.append(np.mean(ave_in_hum))
ave.append(np.mean(ave_in_pre))
# ave.append(np.mean(ave_in_tem))
ave = np.stack(ave)
print(ave)

############ fill missing value with mean value
test_set = []

for i in range(0, total_samples):
    for j, entry in enumerate(samples[i]):
        # fill in the missing value
        if entry == '':
            samples[i, j] = str(ave[j])
    # samples[i].append(0)

    test_set.append(samples[i])

test_set = np.stack(test_set)

############ Standardization
average = [16.47363382, 73.27385588, 983.75472059, 71.73708676, 984.48064706, 0]
sigma = [4.69977465, 17.58225747, 13.07809764, 15.12643684, 7.57291011, 1]
# average = np.mean(test_set.astype(float), axis=0)
# sigma = np.std(test_set.astype(float), axis=0)
test_set = (test_set.astype(float) - average) / sigma
# print(test_set.shape)     # (407, 6)



############ Add 48 validate


# Writing
print('writing...')

with open('./test/test', 'a', encoding='utf-8') as f:
    for _, line in enumerate(samples1):
        f.write(str(line[0]))
        for j in range(1, len(line)):
            f.write(','+str(line[j]))
        f.write('\n')

    for _, line in enumerate(test_set):
        f.write(str(line[0]))
        for j in range(1, len(line)):
            f.write(','+str(line[j]))
        f.write('\n')