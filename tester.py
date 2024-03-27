import os
import torch
import time
import argparse
from torch import nn
import numpy as np
from models.ReadCurrent import ReadCurrent
from models.LSTM import LSTM
from models.CNN_LSTM import CNN_LSTM
from models.Transformer import Transformer
from models.CNN_Transformer import CNN_Transformer


def modified_zscore(signal, mad_threshold=3.5, consistency_correction=1.4826):
	median = np.median(signal)
	dev_from_med = np.array(signal) - median
	mad = np.median(np.abs(dev_from_med))
	mad_score = dev_from_med / (consistency_correction * mad)

	x = np.where(np.abs(mad_score) > mad_threshold)
	x = x[0]

	if len(x) > 0:
		for i in range(len(x)):
			if x[i] == 0:
				mad_score[x[i]] = mad_score[x[i] + 1]
			elif x[i] == len(mad_score) - 1:
				mad_score[x[i]] = mad_score[x[i] - 1]
			else:
				mad_score[x[i]] = (mad_score[x[i] - 1] + mad_score[x[i] + 1]) / 2
	return mad_score


def cut_patchs(signal, seq_length, stride, patch_size):
	split_signal = np.zeros((patch_size, seq_length), dtype="float32")
	for i in range(seq_length):
		split_signal[:, i] = signal[(i*stride):(i*stride)+patch_size]
	return split_signal


def myprint(string, log):
	log.write(string+'\n')
	print(string)


def inference(inputs, model, label, device):
	true_pred, false_pred = 0, 0
	inputs = torch.FloatTensor(np.array(inputs)).to(device)
	outputs = model(inputs)
	outputs = outputs.max(dim=1).indices
	for y in outputs:
		if y == label:
			true_pred += 1
		else:
			false_pred += 1
	return true_pred, false_pred


def test(model, reads, label, batch_size, cut, length,
			  patches, seq_length, stride, patch_size, log, device):
	model.to(device)
	model.eval()
	with torch.no_grad():
		rejected_reads, accepted_reads, batch_count = 0, 0, 0
		true_pred, false_pred = 0, 0
		inputs = []

		start_time = time.time()
		for read in reads:
			if len(read) < cut + length:
				rejected_reads += 1
				continue
			accepted_reads += 1
			read = modified_zscore(read[cut:cut+length])
			if patches:
				read = cut_patchs(read, seq_length, stride, patch_size)
			inputs.append(read)

			if accepted_reads % batch_size == 0 and accepted_reads != 0:
				batch_count += 1
				t, f = inference(inputs, model, label, device)
				true_pred += t
				false_pred += f
				inputs = []

		if len(inputs) > 0:
			batch_count += 1
			t, f = inference(inputs, model, label, device)
			true_pred += t
			false_pred += f

		if label == 1:
			myprint('accepted pos reads: {}, rejected pos reads: {}, TP: {}, FN: {}'.format(
				accepted_reads, rejected_reads, true_pred, false_pred), log)
		else:
			myprint('accepted neg reads: {}, rejected neg reads: {}, TN: {}, FP: {}'.format(
				accepted_reads, rejected_reads, true_pred, false_pred), log)
		total_time = time.time() - start_time
	return true_pred, false_pred, total_time / batch_count


if __name__ == '__main__':
	# Get command arguments
	parser = argparse.ArgumentParser(description="Test model")
	parser.add_argument("--pos_data_folder", '-p', type=str, required=True, help="Path to the positive dataset folder that contains train, valid, test files (.npy)")
	parser.add_argument("--neg_data_folder", '-n', type=str, required=True, help="Path to the negative dataset folder that contains train, valid, test files (.npy)")
	parser.add_argument("--model_state", '-ms', type=str, required=True, help="Path of the model (a pth file)")
	parser.add_argument("--output", '-o', type=str, required=True, help="The output path")
	parser.add_argument("--batch_size", '-b', type=int, default=512, help="Batch size, default 512")
	parser.add_argument("--cut", '-c', type=int, default=1500, help="Electrical signal length to be cut, default 1500")
	parser.add_argument("--length", '-len', type=int, default=3000, help="The length of each signal segment, default 3000")
	parser.add_argument("--patches", '-patches', action='store_true', help="Convert electrical signals into patches, default False")
	parser.add_argument("--seq_length", '-sl', type=int, default=299, help="Sequence length after patch, default 299")
	parser.add_argument("--stride", '-s', type=int, default=10, help="Patch step size, default 10")
	parser.add_argument("--patch_size", '-ps', type=int, default=16, help="The size of patch, default 16")
	parser.add_argument("--gpu_ids", '-g', type=str, default=None, help="Specify the GPU to use, if not specified, use all GPUs or CPU, default None")
	args = parser.parse_args()

	# output file
	log = open(os.path.join(args.output, 'test.txt'), mode='w', encoding='utf-8')

	# Print parameter information
	for arg in vars(args):
		myprint(f"{arg}: {getattr(args, arg)}", log)

	# Load model
	model = ReadCurrent([32, 64, 128, 256, 512], n_fc_neurons=2048, depth=29, shortcut=True)
	# model = LSTM(args.patch_size, 512, 2, True)
	# model = CNN_LSTM(512, 2, True)
	# model = Transformer(args.patch_size, args.seq_length, 512, 2048, 8, 6, 0.1, use_bias=True)
	# model = CNN_Transformer(512, 2048, 8, 6, 0.1, use_bias=True)

	# Use GPU or CPU
	if args.gpu_ids:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	model.to(device)
	myprint(f"Test in {device} {args.gpu_ids}", log)

	# Load model state
	model.load_state_dict(torch.load(args.model_state))

	# Load dataset and testing
	reads = np.load(os.path.join(args.pos_data_folder, "test.npy"), allow_pickle=True)
	myprint(f"load positive test data from {os.path.join(args.pos_data_folder, 'test.npy')}, shape: {reads.shape}", log)
	tp, fn, pos_infer_time = test(model, reads, 1, args.batch_size, args.cut, args.length,
		args.patches, args.seq_length, args.stride, args.patch_size, log, device)
	
	reads = np.load(os.path.join(args.neg_data_folder, "test.npy"), allow_pickle=True)
	myprint(f"load negative test data from {os.path.join(args.neg_data_folder, 'test.npy')}, shape: {reads.shape}", log)
	tn, fp, neg_infer_time = test(model, reads, 0, args.batch_size, args.cut, args.length,
		args.patches, args.seq_length, args.stride, args.patch_size, log, device)

	# Calculate evaluation index values
	accuracy = round((tp + tn) * 100 / (tp + tn + fp + fn), 2)
	precision = round(tp * 100 / (tp + fp), 2)
	recall = round(tp * 100 / (tp + fn), 2)
	f1_score = round((2 * precision * recall) / (precision + recall), 2)
	aver_infer_time = round((pos_infer_time + neg_infer_time) / 2, 4)
	myprint(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1_score: {f1_score}, average inference time: {aver_infer_time}", log)
	log.close()