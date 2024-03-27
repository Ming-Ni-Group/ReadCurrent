import torch
from torch import nn
import torch.nn.functional as F


class CNN_LSTM(nn.Module):
	def __init__(self, num_hiddens, num_layers=1, bidirectional=False, **kwargs):
		super(CNN_LSTM, self).__init__(**kwargs)
		self.conv = nn.Conv1d(1, 32, kernel_size=20, padding=5, stride=5)
		self.bn = nn.BatchNorm1d(32)
		self.relu = nn.ReLU()
		self.pool = nn.MaxPool1d(5, padding=2, stride=3)

		self.lstm = nn.LSTM(input_size=32, hidden_size=num_hiddens, num_layers=num_layers, bias=True, bidirectional=bidirectional)
		self.linear = nn.Linear(num_hiddens * 2, 2)

		# initialization
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight)
				nn.init.constant_(m.bias, 0)
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm1d)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, (nn.LayerNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, input):
		input = input.unsqueeze(1)
		output = self.conv(input)
		output = self.bn(output)
		output = self.relu(output)
		output = self.pool(output)

		# [batch_size, hidden_size, seq_len] => [seq_len, batch_size, hidden_size]
		output = output.transpose(1, 2).transpose(0, 1)
		output, (hn, cn) = self.lstm(output)
		forward_hn = hn[-2]
		backward_hn = hn[-1]
		output = torch.cat((forward_hn, backward_hn), dim=1)
		# X = torch.flatten(X, start_dim=1)
		# output = self.linear(output)
		output = F.softmax(self.linear(output), dim=1)
		return output
	

if __name__ == '__main__':
	input = torch.randn(128, 3000)
	model = CNN_LSTM(1024, 1, True)
	output = model(input)