import os
import argparse
import numpy as np


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


def train_normalization(data, cut, length, tile, patches, seq_length, stride, patch_size):
	step, start, segment_arr = length // tile, 0, []
	for _ in range(tile):
		for signal in data:
			if len(signal) < cut + length:
				continue

			signal = signal[cut:]
			end = start + length
			while end <= len(signal):
				segment = modified_zscore(signal[end-length:end])
				if patches:
					segment = cut_patchs(segment, seq_length, stride, patch_size)
				segment_arr.append(segment)
				end += length
		start += step
	return np.array(segment_arr)


def valid_normalization(data, cut, length, patches, seq_length, stride, patch_size):
	segment_arr = []
	for signal in data:
		if len(signal) < cut + length:
			continue
		segment = modified_zscore(signal[cut:cut+length])
		if patches:
			segment = cut_patchs(segment, seq_length, stride, patch_size)
		segment_arr.append(segment)
	return np.array(segment_arr)


if __name__ == '__main__':
	# Get command arguments
	parser = argparse.ArgumentParser(description="Data preprocessing")
	parser.add_argument("--data_folder", '-d', type=str, required=True, help="Path to the dataset folder that contains train, valid, test files (.npy)")
	parser.add_argument("--cut", '-c', type=int, default=1500, help="Electrical signal length to be cut, default 1500")
	parser.add_argument("--tiling_fold", '-tf', type=int, default=3, help="Number of tiles, default 3")
	parser.add_argument("--length", '-l', type=int, default=3000, help="The length of the sliding window, default 3000")
	parser.add_argument("--patches", '-patches', action='store_true', help="Convert electrical signals into patches, default False")
	parser.add_argument("--seq_length", '-sl', type=int, default=299, help="Sequence length after patch, default 299")
	parser.add_argument("--stride", '-s', type=int, default=10, help="Patch step size, default 10")
	parser.add_argument("--patch_size", '-ps', type=int, default=16, help="The size of patch, default 16")
	args = parser.parse_args()

	# Print parameter information
	for arg in vars(args):
		print(f"{arg}: {getattr(args, arg)}")

	# Load dataset
	print("\nLoad dataset!")
	train_data = np.load(os.path.join(args.data_folder, "train.npy"), allow_pickle=True)
	print(f"Successfully loaded training set from {os.path.join(args.data_folder, 'train.npy')}, shape: {train_data.shape}")
	valid_data = np.load(os.path.join(args.data_folder, "valid.npy"), allow_pickle=True)
	print(f"Successfully loaded validation set from {os.path.join(args.data_folder, 'valid.npy')}, shape: {valid_data.shape}")

	# Normalize using modified Z-score and cut signal
	print("\nPreprocess dataset!")
	train_data = train_normalization(train_data, args.cut, args.length, args.tiling_fold,
								  args.patches, args.seq_length, args.stride, args.patch_size)
	print(f"Successfully preprocessed training set, shape: {train_data.shape}")
	valid_data = valid_normalization(valid_data, args.cut, args.length,
								  args.patches, args.seq_length, args.stride, args.patch_size)
	print(f"Successfully preprocessed validation set, shape: {valid_data.shape}")

	# Create training data and validation data (npy)
	print("\nSave data!")
	np.save(os.path.join(args.data_folder, "train_preprocessed.npy"), train_data)
	print(f"Successfully saved preprocessed training set to {os.path.join(args.data_folder, 'train_preprocessed.npy')}")
	np.save(os.path.join(args.data_folder, "valid_preprocessed.npy"), valid_data)
	print(f"Successfully saved preprocessed validation set to {os.path.join(args.data_folder, 'valid_preprocessed.npy')}")