# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
import argparse
from RNNModel import RNNModel



def get_train_validate(train_path):
	pass

def get_test(train_path, test_path):
	pass

def summit_test(output):
	f = open('./test/test.csv', encoding='utf-8')
	f_content = f.readlines()
	f.close()

	samples = []

	for i, line in enumerate(f_content):
		if i >= 1:
			current_line_split = line.strip().split(',')
			samples.append(current_line_split[0])

	samples = np.stack(samples)

	with open('./test/summit.csv', 'w+', encoding='utf-8') as f:
		f.write('time,temperature')
		f.write('\n')
		for i, line in enumerate(output):
			f.write(str(samples[i]))
			f.write(',' + str(line))
			f.write('\n')

# used for generating training data and validation data
def data_generator(data, look_back, future_step, batch_size, shuffle=False):
	# look_back: 48
	# batch_size: 128
	start = 0 + look_back
	end = len(data)  # [start, end)
	# end = len(data) - future_step # [start, end)
	# start += 128;	end not change
	while True:
		if shuffle:
			rows = np.random.randint(start, end, size=batch_size)
		else:
			# len(rows): 128 or (end - start)
			rows = range(start, min(start + batch_size, end))
			# back to start
			if start + batch_size >= end:
				start = 0 + look_back
			else:
				start += batch_size

		batch = np.zeros([len(rows), look_back, data.shape[-1]])
		target = np.zeros([len(rows),])
		for j, row in enumerate(rows):
			### key
			indices = range(rows[j]-look_back, rows[j])	# range(48-48, 48) = range(0, 48) = [0, 47]
			batch[j] = data[indices]
			target[j] = data[rows[j]][-1]
		yield np.swapaxes(batch, 0, 1)[:, :, 0:5], target
		# (len, batch, feature):(48, 128, 5)
		# yield np.swapaxes(batch, 0, 1), target
# Please revise these function for generating submission
def test_model(model, test_XY, device, num_test_batch, epoch, sigma, average):
	output = np.array([])
	target = np.array([])
	for batch in range(num_test_batch):
		test_X, test_Y = next(test_XY)
		test_X = torch.tensor(test_X, dtype=torch.float32, device=device)
		state = None
		Y_predict, _ = model(test_X, state)
		output = np.append(output, np.array(Y_predict.detach().cpu(), dtype=float))
		target = np.append(target, test_Y)

	if epoch == -1:
		print('len of output:', len(output))
		print()
		print('###Predicted Temperature:', str(output))
		summit_test(output)
		return 0
	# output = output * sigma + average
	# target = target * sigma + average

	# calculate MSE
	MSE = np.mean(np.square(output - target))
	
	if epoch == 199:
		print('###The predicted temperature in epoch %s are: %s'%(epoch, str(output)))
	print('###The MSE in epoch %s is: %.6f'%(epoch, MSE))
	print('\n')
	return MSE



def train_test_model(model, train_XY, test_XY, num_train_batch, num_test_batch, num_epoch, lr,
				device, pre_step, sigma, average):
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	model.to(device)

	# model training
	print('training...')
	for epoch in range(num_epoch):
		
		for batch in range(num_train_batch):
			train_X, train_Y = next(train_XY)
			train_X = torch.tensor(train_X, dtype=torch.float32, device=device)
			state = None
			# train_X: (48, 128, 5)/(len, batch, feature)
			# output: 128
			output, _ = model(train_X, state)
			train_Y = torch.tensor(train_Y, dtype=torch.float32, device=device)
			# # Add
			# train_Y = train_Y * sigma + average

			loss = criterion(output, train_Y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		
		# for X, Y in inputs:
		# 	if state is not None: # detach the variable from the graph
		# 		if isinstance(state, tuple): # LSTM, state: (h, c)
		# 			state = (state[0].detach(), state[1].detach())
		# 		else:
		# 			state = state.detach()
		# 	output, state = model(X, state)
		# 	Y_predict = torch.tensor(Y)
		# 	loss = criterion(output, Y_predict)
		# 	optimizer.zero_grad() # set the gradient of last step to zero
		# 	loss.backward() # compute the gradient of the parameters
		# 	optimizer.step() # update the parameters

		print('###Training loss in epoch %s: %.6f'%(epoch, loss.item()))
		print('-------------')
		if (epoch + 1) % pre_step == 0:
			MSE = test_model(model, test_XY, device, num_test_batch, epoch, sigma, average)
			# if epoch > 100 and MSE < 0.15:
			# 	break
	

def parse():
	parser = argparse.ArgumentParser(description='Predicting temperature.')
	parser.add_argument('--hidden_size', type=int, default=8, 
		help='number of neurons in the hidden layer')
	parser.add_argument('--num_feature', type=int, default=5,
		help='number of features of the input')
	parser.add_argument('--batch_size', type=int, default=128,
		help='batch size in each batch')
	# parser.add_argument('--sql', type=int, default=48,
	# 	help='length of the sequence')
	parser.add_argument('--num_epoch', type=int, default=200,
		help='number of epoch')
	parser.add_argument('--lr', type=float, default=0.01,
		help='learning rate')
	parser.add_argument('--pre_step', type=int, default=10,
		help='steps interval to print/save the testing result')
	parser.add_argument('--look_back', type=int, default=48,
		help='number of samples in the past used to predict the temperature')
	parser.add_argument('--future_step', type=int, default=1,
		help='which future time stamp should the model predict')
	parser.add_argument('--train_path', type=str, default='./train/train_validate',
		help='the path of the file for generating training set')
	parser.add_argument('--validate_path', type=str, default='./train/validate',
		help='the path of the file for generating validating set')
	parser.add_argument('--test_path', type=str, default='./test/test',
		help='the path of the file for generating testing set')
	parser.add_argument('--is_test', type=bool, default=False,
		help='whether use the test set to test the model')
	parser.add_argument('--shuffle_train', type=bool, default=True,
		help='whether to randomly generate the data for training')
	parser.add_argument('--shuffle_validate', type=bool, default=False,
		help='whether to randomly generate the data for validating')
	parser.add_argument('--shuffle_test', type=bool, default=False,
		help='whether to randomly generate the data for testing')
	parser.add_argument('--average', type=float, default=16.534820588235302,
		help='return to real temperature')
	parser.add_argument('--sigma', type=float, default=4.071784519555742,
		help='return to real temperature')

	return parser.parse_args()


def main():
	parser = parse() # get the parameters
	hidden_size = parser.hidden_size
	num_feature = parser.num_feature
	batch_size = parser.batch_size
	# sql = parser.sql
	num_epoch = parser.num_epoch
	lr = parser.lr
	pre_step = parser.pre_step
	look_back = parser.look_back
	future_step = parser.future_step
	train_path = parser.train_path
	test_path = parser.test_path
	validate_path = parser.validate_path
	is_test = parser.is_test
	shuffle_train = parser.shuffle_train
	shuffle_validate = parser.shuffle_validate
	shuffle_test = parser.shuffle_test

	average = parser.average
	sigma = parser.sigma

	### generate the training set, validating set and test set
	train_set = []
	validate_set = []
	test_set = []
	with open(train_path, encoding='utf-8') as f:
		content = f.readlines()
		for line in content:
			line_tuple = line.strip().split(',')
			train_set.append(np.array(line_tuple, dtype=float))
	train_set = np.stack(train_set)

	with open(validate_path, encoding='utf-8') as f:
		content = f.readlines()
		for line in content:
			line_tuple = line.strip().split(',')
			validate_set.append(np.array(line_tuple, dtype=float))
	validate_set = np.stack(validate_set)

	with open(test_path, encoding='utf-8') as f:
		content = f.readlines()
		for line in content:
			line_tuple = line.strip().split(',')
			test_set.append(np.array(line_tuple, dtype=float))
	test_set = np.stack(test_set)

	### define model
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	rnn = nn.GRU(num_feature, hidden_size)
	model = RNNModel(rnn, num_feature)
	model.to(device)


	train_XY = data_generator(train_set, look_back, future_step, batch_size, shuffle_train)
	validate_XY = data_generator(validate_set, look_back, future_step, batch_size, shuffle_validate)
	test_XY = data_generator(test_set, look_back, future_step, batch_size, shuffle_test)
	print(test_set.shape)

	num_train_batch = (len(train_set) - look_back - future_step) // batch_size + 1
	num_validate_batch = (len(validate_set) - look_back - future_step) // batch_size + 1
	num_test_batch = (len(test_set) - look_back - future_step) // batch_size + 1

	### train
	train_test_model(model, train_XY, validate_XY, num_train_batch, num_validate_batch,
					num_epoch, lr, device, pre_step, sigma, average)
	### predict
	average = [18.07636863, 71.85216879, 982.82304936, 70.41442357, 980.7735828, 0]
	sigma = [5.23379147, 17.51692138, 9.954381, 15.24855444, 33.68555629, 1]
	test_model(model, test_XY, device, num_test_batch, -1, sigma, average)



if __name__ == '__main__':
	main()