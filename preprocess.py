# -*- coding: utf-8 -*-

import numpy as np

'''
train.csv and test.csv file
the header of these two files are:
time,年,月,日,小时,分钟,秒,温度(室外),湿度(室外),气压(室外),湿度(室内),气压(室内),temperature
totally 13 columns
'''
f = open('./train/train.csv', encoding='utf-8')

# CountInc = 0 # count the number of lines that have incomplete information, e.g. '湿度(室外)' is absent
# CountWrong = 0 # count the number of lines that have wrong statiscis, expecially the wrong '气压(室内)''

f_content = f.readlines()
f.close()

total_samples = len(f_content)-1

train_percentage = 0.8

# validate_percentage = 0.2

boundary = int(total_samples * train_percentage - 1)

samples = []

for i, line in enumerate(f_content):
	if i >= 1:
		current_line_split = line.strip().split(',')
		samples.append(current_line_split[-6:])

samples = np.stack(samples)

sample_steps = 30

train_set = []

validate_set = []

############## calculate the average value of the features to get rid of the outliers
outdoor_tem = samples[:, 0]
outdoor_humidity = samples[:, 1]
outdoor_pressure = samples[:, 2]
indoor_humidity = samples[:, 3]
indoor_pressure = samples[:, 4]
indoor_tem = samples[:, 5]

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

ave_out_tem = np.mean(ave_out_tem)
ave_out_hum = np.mean(ave_out_hum)
ave_out_pre = np.mean(ave_out_pre)
ave_in_hum = np.mean(ave_in_hum)
ave_in_pre = np.mean(ave_in_pre)
ave_in_tem = np.mean(ave_in_tem)

print(ave_out_tem, ave_out_hum, ave_out_pre, ave_in_hum, ave_in_pre, ave_in_tem)

# transform the outlier into ''
for i in range(len(outdoor_tem)):
	if outdoor_tem[i] != '' and abs(float(outdoor_tem[i]) - ave_out_tem) > 100:
		outdoor_tem[i] = ''
	if outdoor_humidity[i] != '' and abs(float(outdoor_humidity[i]) - ave_out_hum) > 100:
		outdoor_humidity[i] = ''
	if outdoor_pressure[i] != '' and abs(float(outdoor_pressure[i]) - ave_out_pre) > 100:
		outdoor_pressure[i] = ''
	if indoor_humidity[i] != '' and abs(float(indoor_humidity[i]) - ave_in_hum) > 100:
		indoor_humidity[i] = ''
	if indoor_pressure[i] != '' and abs(float(indoor_pressure[i]) - ave_in_pre) > 100:
		indoor_pressure[i] = ''
	if indoor_tem[i] != '' and abs(float(indoor_tem[i]) - ave_in_tem) > 100:
		indoor_tem[i] = ''

samples[:, 0] = outdoor_tem
samples[:, 1] = outdoor_humidity
samples[:, 2] = outdoor_pressure
samples[:, 3] = indoor_humidity
samples[:, 4] = indoor_pressure
samples[:, 5] = indoor_tem


############# samples every 30min, fill in the missing value using the mean value 
############# of the neighbors, i.e., 2h's neighbors (120min)
############# One can try to sample from end to start, i.e. range(total_samples, 0, sample_steps)
for i in range(0, total_samples, sample_steps):
	for j, entry in enumerate(samples[i]):
		# fill in the missing value
		if entry == '':
			if i < 60:
				neighbors = samples[i+1:i+120, j]
				neighbors = np.delete(neighbors, np.where(neighbors == ''))
				neighbors = np.array(neighbors, dtype=float)
				mean_value = np.mean(neighbors)
				samples[i, j] = str(mean_value)
			elif i >= (total_samples - 60):
				neighbors = samples[i-120:i-1, j]
				neighbors = np.delete(neighbors, np.where(neighbors == ''))
				neighbors = np.array(neighbors, dtype=float)
				mean_value = np.mean(neighbors)
				samples[i, j] = str(mean_value)
			else:
				neighbors = np.append(samples[i-60:i-1, j], samples[i+1:i+60, j])
				neighbors = np.delete(neighbors, np.where(neighbors == ''))
				neighbors = np.array(neighbors, dtype=float)
				mean_value = np.mean(neighbors)
				samples[i, j] = str(mean_value)

	if i <= boundary: # lie in the train set
		train_set.append(samples[i])
	else:
		validate_set.append(samples[i])

train_set = np.stack(train_set)
validate_set = np.stack(validate_set)

print('here!')
# Add standardization
average = np.mean(train_set, axis=0)
sigma = np.std(train_set, axis=0)
train_set = (train_set - average) / sigma

print('writing...')


with open('./train/train0', 'a', encoding='utf-8') as f:
	for _, line in enumerate(train_set):
		f.write(line[0])
		for j in range(1, len(line)):
			f.write(','+line[j])
		f.write('\n')

# with open('./train/validate0', 'a', encoding='utf-8') as f:
# 	for _, line in enumerate(validate_set):
# 		f.write(line[0])
# 		for j in range(1, len(line)):
# 			f.write(','+line[j])
# 		f.write('\n')
