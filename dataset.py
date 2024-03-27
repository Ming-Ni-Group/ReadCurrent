import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
	def __init__(self, data, data_type='pos'):
		self.data = data
		if data_type == 'pos':
			self.label = np.ones(self.data.shape[0])
		else:
			self.label = np.zeros(self.data.shape[0])

	def __len__(self):
		return len(self.label)

	def __getitem__(self, index):
		X = self.data[index]
		Y = self.label[index]
		return X, Y