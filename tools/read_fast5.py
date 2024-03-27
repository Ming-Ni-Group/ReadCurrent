import os
import argparse
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file


def read_fast5(file_dir, min_length, size, ids=None):
	file_count, reads_count, total_len = 0, 0, 0
	accepted_reads, rejected_reads, short_reads = 0, 0, 0
	reads_array = []

	for root, _, files in os.walk(file_dir):
		print("current folder: ", root)
		for file in files:
			if not file.lower().endswith('.fast5'):
				continue
			file_count += 1
			file_path = os.path.join(root, file)
			f5 = get_fast5_file(file_path, mode='r')

			for read_id in f5.get_read_ids():
				reads_count += 1
				if ids and read_id not in ids:
					rejected_reads += 1
					print(f"fast5 files count {file_count}, reads count {reads_count}, accepted reads {accepted_reads}, rejected reads {rejected_reads}, short reads {short_reads}")
					continue
				read = f5.get_read(read_id)
				signal = read.get_raw_data()
				
				if len(signal) < min_length:
					short_reads += 1
					print(f"fast5 files count {file_count}, reads count {reads_count}, accepted reads {accepted_reads}, rejected reads {rejected_reads}, short reads {short_reads}")
					continue

				accepted_reads += 1
				total_len += len(signal)
				reads_array.append(signal)
				print(f"fast5 files count {file_count}, reads count {reads_count}, accepted reads {accepted_reads}, rejected reads {rejected_reads}, short reads {short_reads}")

				if accepted_reads == size:
					return np.array(reads_array, dtype=object)
					
	print('The number of accepted reads less than dataset size!')
	exit(0)


def save_reads(reads, output):
	np.random.shuffle(reads)
	train_data = reads[:args.train_size]
	valid_data = reads[args.train_size:args.train_size + args.valid_size]
	test_data = reads[args.train_size + args.valid_size:]
	np.save(os.path.join(output, 'train.npy'), train_data)
	print(f"train data saved to {os.path.join(output, 'train.npy')}, shape: {train_data.shape}")
	np.save(os.path.join(output, 'valid.npy'), valid_data)
	print(f"valid data saved to {os.path.join(output, 'valid.npy')}, shape: {valid_data.shape}")
	np.save(os.path.join(output, 'test.npy'), test_data)
	print(f"test data saved to {os.path.join(output, 'test.npy')}, shape: {test_data.shape}")
	exit(0)


if __name__ == '__main__':
	# Get command arguments
	parser = argparse.ArgumentParser(description="Read fast5")
	parser.add_argument("--file_dir", '-dir', type=str, required=True, help="The directory where the fast5 files is located")
	parser.add_argument("--output", '-o', type=str, required=True, help="Storage path for output files")
	parser.add_argument("--read_ids", '-ids', type=str, default=None, help="The path for read ids file")
	parser.add_argument("--min_length", '-len', type=int, default=4500, help="Minimum length of each electrical signal, default 4500")
	parser.add_argument("--train_size", '-train', type=int, default=20000, help="Number of electrical signals to be read for training, default 20000")
	parser.add_argument("--valid_size", '-valid', type=int, default=10000, help="Number of electrical signals to be read for validation, default 10000")
	parser.add_argument("--test_size", '-test', type=int, default=10000, help="Number of electrical signals to be read for testing, default 10000")
	args = parser.parse_args()

	# Print parameter information
	for arg in vars(args):
		print(f"{arg}: {getattr(args, arg)}")

	if not os.path.exists(args.output):
		os.makedirs(args.output)

	ids = None
	if args.read_ids is not None:
		ids_file = open(args.read_ids, 'r')
		ids = [id.strip() for id in ids_file.readlines()]

	total_size = args.train_size + args.valid_size + args.test_size
	signals = read_fast5(args.file_dir, args.min_length, total_size, ids)
	save_reads(signals, args.output)