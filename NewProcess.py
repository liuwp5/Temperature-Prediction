import numpy as np

###### read train.csv
train_f = open('./train/train.csv', encoding='utf-8')
train_content = train_f.readlines()
train_f.close()

train_total_samples = len(train_content)-1

all_samples = []
train_samples = []
test_samples = []

for i, line in enumerate(train_content):
    if i >= 1:
        current_line_split = line.strip().split(',')
        train_samples.append(current_line_split[-6:])
        # all_samples.append(current_line_split[-6:])

train_samples = np.stack(train_samples)

###### read test.csv
test_f = open('./test/test.csv', encoding='utf-8')
test_content = test_f.readlines()
test_f.close()

test_total_samples = len(test_content)-1


for i, line in enumerate(test_content):
    if i >= 1:
        current_line_split = line.strip().split(',')
        current_line_split.append(0)
        test_samples.append(current_line_split[-6:])
        # all_samples.append(current_line_split[-6:])

test_samples = np.stack(test_samples)
# all_samples = np.stack(all_samples)

##############################################################################


###### calculate the average value of the features to get rid of the outliers
outdoor_tem = train_samples[:, 0]
outdoor_humidity = train_samples[:, 1]
outdoor_pressure = train_samples[:, 2]
indoor_humidity = train_samples[:, 3]
indoor_pressure = train_samples[:, 4]
# indoor_tem = train_samples[:, 5]

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

ave_out_tem = np.mean(ave_out_tem)
ave_out_hum = np.mean(ave_out_hum)
ave_out_pre = np.mean(ave_out_pre)
ave_in_hum = np.mean(ave_in_hum)
ave_in_pre = np.mean(ave_in_pre)

print(ave_out_tem, ave_out_hum, ave_out_pre, ave_in_hum, ave_in_pre)

###### transform the outlier into ''
for i in range(len(outdoor_tem)):
    if outdoor_tem[i] != '' and abs(float(outdoor_tem[i]) - ave_out_tem) > 50:
        outdoor_tem[i] = ''
    if outdoor_humidity[i] != '' and abs(float(outdoor_humidity[i]) - ave_out_hum) > 50:
        outdoor_humidity[i] = ''
    if outdoor_pressure[i] != '' and abs(float(outdoor_pressure[i]) - ave_out_pre) > 50:
        outdoor_pressure[i] = ''
    if indoor_humidity[i] != '' and abs(float(indoor_humidity[i]) - ave_in_hum) > 50:
        indoor_humidity[i] = ''
    if indoor_pressure[i] != '' and abs(float(indoor_pressure[i]) - ave_in_pre) > 50:
        indoor_pressure[i] = ''

train_samples[:, 0] = outdoor_tem
train_samples[:, 1] = outdoor_humidity
train_samples[:, 2] = outdoor_pressure
train_samples[:, 3] = indoor_humidity
train_samples[:, 4] = indoor_pressure


train_validate_set = []
test_set = []
all_set = []

###### samples every 30min, fill in the missing value using the mean value
sample_steps = 30
for i in range(0, train_total_samples, sample_steps):
    for j, entry in enumerate(train_samples[i]):
        # fill in the missing value
        if entry == '':
            if i < 60:
                neighbors = train_samples[i+1:i+120, j]
                neighbors = np.delete(neighbors, np.where(neighbors == ''))
                neighbors = np.array(neighbors, dtype=float)
                mean_value = np.mean(neighbors)
                train_samples[i, j] = str(mean_value)
            elif i >= (train_total_samples - 60):
                neighbors = train_samples[i-120:i-1, j]
                neighbors = np.delete(neighbors, np.where(neighbors == ''))
                neighbors = np.array(neighbors, dtype=float)
                mean_value = np.mean(neighbors)
                train_samples[i, j] = str(mean_value)
            else:
                neighbors = np.append(train_samples[i-60:i-1, j], train_samples[i+1:i+60, j])
                neighbors = np.delete(neighbors, np.where(neighbors == ''))
                neighbors = np.array(neighbors, dtype=float)
                mean_value = np.mean(neighbors)
                train_samples[i, j] = str(mean_value)
    # print('i:', i)
    train_validate_set.append(train_samples[i])
train_validate_set = np.stack(train_validate_set)

###### test
###### calculate the average value of the features
outdoor_tem = test_samples[:, 0]
outdoor_humidity = test_samples[:, 1]
outdoor_pressure = test_samples[:, 2]
indoor_humidity = test_samples[:, 3]
indoor_pressure = test_samples[:, 4]
indoor_tem = test_samples[:, 5]

ave_out_tem, ave_out_hum, ave_out_pre, ave_in_hum, ave_in_pre, ave_in_tem = [], [], [], [], [], []

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
    if indoor_tem[i] != '':
        ave_in_tem.append(float(indoor_tem[i]))

ave = []
ave.append(np.mean(ave_out_tem))
ave.append(np.mean(ave_out_hum))
ave.append(np.mean(ave_out_pre))
ave.append(np.mean(ave_in_hum))
ave.append(np.mean(ave_in_pre))
ave.append(np.mean(ave_in_tem))
ave = np.stack(ave)
print('test_ave:', ave)

###### fill missing value with mean value
for i in range(0, test_total_samples):
    for j, entry in enumerate(test_samples[i]):
        # fill in the missing value
        if entry == '' or abs(float(entry) - ave[j]) > 50:
            # test_samples[i, j] = str(ave[j])
            if i < 5:
                neighbors = test_samples[i + 1:i + 10, j]
                neighbors = np.delete(neighbors, np.where(neighbors == ''))
                neighbors = np.array(neighbors, dtype=float)
                mean_value = np.mean(neighbors)
                test_samples[i, j] = str(mean_value)
            elif i >= (train_total_samples - 5):
                neighbors = test_samples[i - 10:i - 1, j]
                neighbors = np.delete(neighbors, np.where(neighbors == ''))
                neighbors = np.array(neighbors, dtype=float)
                mean_value = np.mean(neighbors)
                test_samples[i, j] = str(mean_value)
            else:
                neighbors = np.append(test_samples[i - 5:i - 1, j], test_samples[i + 1:i + 5, j])
                neighbors = np.delete(neighbors, np.where(neighbors == ''))
                neighbors = np.array(neighbors, dtype=float)
                mean_value = np.mean(neighbors)
                test_samples[i, j] = str(mean_value)
    # samples[i].append(0)

    test_set.append(test_samples[i])
test_set = np.stack(test_set)

# print(train_validate_set.shape) # (850, 6)
# print(test_set.shape)   # (406, 6)

###### stack train_validate_set and test_set
all_set = np.vstack((train_validate_set, test_set))
# print(all_set.shape)
# print(all_set)

###### Standardization
average = np.mean(all_set.astype(float), axis=0)
sigma = np.std(all_set.astype(float), axis=0)
average[5] = 0
sigma[5] = 1
all_set = (all_set.astype(float) - average) / sigma

print('average:', average)
print('sigma:', sigma)

###### get files
train_validate_length = len(train_validate_set)
test_length = len(test_set)

train_validate = all_set[0:train_validate_length]
train = train_validate[0:int(0.8*train_validate_length)]
validate = train_validate[int(0.8*train_validate_length):train_validate_length]
test = all_set[train_validate_length:]
# print(train_validate.shape)
# print(train.shape)
# print(validate.shape)
# print(test.shape)

###### writing files
print('writing...')
with open('./train/train', 'w+', encoding='utf-8') as f:
    for _, line in enumerate(train):
        f.write(str(line[0]))
        for j in range(1, len(line)):
            f.write(','+str(line[j]))
        f.write('\n')

with open('./train/validate', 'w+', encoding='utf-8') as f:
    for _, line in enumerate(validate):
        f.write(str(line[0]))
        for j in range(1, len(line)):
            f.write(','+str(line[j]))
        f.write('\n')

with open('./train/train_validate', 'w+', encoding='utf-8') as f:
    for _, line in enumerate(train_validate):
        f.write(str(line[0]))
        for j in range(1, len(line)):
            f.write(','+str(line[j]))
        f.write('\n')

with open('./test/test', 'w+', encoding='utf-8') as f:
    ############ Add 48 validate
    validate_48 = validate[-48:]
    for _, line in enumerate(validate_48):
        f.write(str(line[0]))
        for j in range(1, len(line)):
            f.write(','+str(line[j]))
        f.write('\n')

    for _, line in enumerate(test):
        f.write(str(line[0]))
        for j in range(1, len(line)):
            f.write(','+str(line[j]))
        f.write('\n')
