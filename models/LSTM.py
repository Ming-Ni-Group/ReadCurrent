import torch
from torch import nn
import torch.nn.functional as F


class LSTM(nn.Module):
	def __init__(self, input_size, num_hiddens, num_layers=1, bidirectional=False, **kwargs):
		super(LSTM, self).__init__(**kwargs)

		self.lstm = nn.LSTM(input_size=input_size, hidden_size=num_hiddens, num_layers=num_layers, bias=True, bidirectional=bidirectional)
		self.linear = nn.Linear(num_hiddens * 2, 2)
		self.relu = nn.ReLU()

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
		self.lstm.flatten_parameters()
		# [batch_size, hidden_size, seq_len] => [seq_len, batch_size, hidden_size]
		input = input.transpose(1, 2).transpose(0, 1)
		output, (hn, cn) = self.lstm(input)
		forward_hn = hn[-2]
		backward_hn = hn[-1]
		output = torch.cat((forward_hn, backward_hn), dim=1)
		# X = torch.flatten(X, start_dim=1)
		# X = self.linear(X)
		output = F.softmax(self.linear(output), dim=1)
		return output


if __name__ == '__main__':
	input = torch.randn(128, 32, 290)
	model = LSTM(32, 1024, 2, True)
	output = model(input)