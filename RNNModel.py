import numpy as np
import torch
from torch import nn

class RNNModel(nn.Module):
	"""docstring for RNNModel"""
	def __init__(self, rnn_layer, num_feature):
		super(RNNModel, self).__init__()
		self.rnn_layer = rnn_layer
		self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
		self.num_feature = num_feature
		self.state = None
		self.dense = nn.Linear(self.hidden_size, 1) # output layer, output the pred


	def forward(self, inputs, state):
		#inputs: look_back*batch_size*num_feature
		# Y: look_back*batch_size*hidden_size
		# state: 1*batch_size*hidden_size
		Y, self.state = self.rnn_layer(inputs, state)
		# use the output of the last step to predict
		output = self.dense(torch.transpose(Y, 0, 1))[:,-1,:].view(inputs.shape[1],) # batch_size * 1
		return output, self.state

