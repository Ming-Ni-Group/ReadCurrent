import argparse
import functools
import logging
from multiprocessing.pool import ThreadPool
from pathlib import Path
import time
import typing

import grpc
import numpy as np

import read_until

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import csv
import torch
from torch import nn
from models.ReadCurrent import ReadCurrent


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


def get_parser() -> argparse.ArgumentParser:
	"""Build argument parser for example"""
	parser = argparse.ArgumentParser("Read until API demonstration..")
	parser.add_argument("--host", default="127.0.0.1", help="MinKNOW server host.")
	parser.add_argument(
		"--port", type=int, default=8000, help="MinKNOW gRPC server port."
	)
	parser.add_argument(
		"--ca-cert",
		type=Path,
		default=None,
		help="Path to alternate CA certificate for connecting to MinKNOW.",
	)
	parser.add_argument("--workers", default=1, type=int, help="worker threads.")
	parser.add_argument(
		"--analysis_delay",
		type=int,
		default=1,
		help="Period to wait before starting analysis.",
	)
	parser.add_argument(
		"--run_time", type=int, default=30, help="Period to run the analysis."
	)
	parser.add_argument(
		"--unblock_duration",
		type=float,
		default=0.1,
		help="Time (in seconds) to apply unblock voltage.",
	)
	parser.add_argument(
		"--one_chunk",
		default=False,
		action="store_true",
		help="Minimum read chunk size to receive.",
	)
	parser.add_argument(
		"--min_chunk_size",
		type=int,
		default=2000,
		help="Minimum read chunk size to receive. NOTE: this functionality "
		"is currently disabled; read chunks received will be unfiltered.",
	)
	parser.add_argument(
		"--debug",
		help="Print all debugging information",
		action="store_const",
		dest="log_level",
		const=logging.DEBUG,
		default=logging.WARNING,
	)
	parser.add_argument(
		"--verbose",
		help="Print verbose messaging.",
		action="store_const",
		dest="log_level",
		const=logging.INFO,
	)
	parser.add_argument(
		"--model_state",
		type=str,
		required=True,
		help="Path of the model (a pth file)",
	)
	parser.add_argument(
		"--output",
		type=str,
		required=True,
		help="The output path",
	)
	parser.add_argument(
		"--gpu_ids",
		type=str,
		default=None,
		help="Specify the GPU to use, if not specified, use all GPUs or CPU, default None",
	)
	return parser


def simple_analysis(
	model,
	device,
	client: read_until.ReadUntilClient,
	output: str,
	batch_size: int = 512,
	delay: float = 1,
	throttle: float = 0.1,
	unblock_duration: float = 0.1,
):
	"""A simple demo analysis leveraging a `ReadUntilClient` to manage
	queuing and expiry of read data.

	:param client: an instance of a `ReadUntilClient` object.
	:param batch_size: number of reads to pull from `client` at a time.
	:param delay: number of seconds to wait before starting analysis.
	:param throttle: minimum interval between requests to `client`.
	:param unblock_duration: time in seconds to apply unblock voltage.

	"""
	logger = logging.getLogger("Analysis")
	logger.warning(
		"Initialising simple analysis. "
		"This will likely not achieve anything useful. "
		"Enable --verbose or --debug logging to see more."
	)
	# we sleep a little simply to ensure the client has started initialised
	logger.info("Starting analysis of reads in %ss.", delay)
	time.sleep(delay)

	sampling_file = open(os.path.join(output,'adaptive_sampling.csv'), mode='w', newline='')
	sampling_writer = csv.writer(sampling_file)
	sampling_writer.writerow(['batch_time', 'read_number', 'channel', 'read_id', 'num_samples', 'decision'])
	target_counter, non_target_counter, short_counter, control_counter = 0, 0, 0, 0

	while client.is_running:
		time_begin = time.time()
		# get the most recent read chunks from the client
		read_batch = client.get_read_chunks(batch_size=batch_size, last=True)
		read_list, inputs = [], []

		for channel, read in read_batch:
			# convert the read data into a numpy array of correct type
			raw_data = np.frombuffer(read.raw_data, client.signal_dtype)
			read.raw_data = bytes("", "utf8")
			# raw_data = raw_data.astype("float32")
			signal_length = len(raw_data)

			# 257-512 channels as control group
			if channel > 256:
				control_counter += 1
				client.stop_receiving_read(channel, read.number)
				row = [time_begin, read.number, channel, read.id, signal_length, 'control']
				sampling_writer.writerow(row)
				continue

			# our data preprocessing operations
			if signal_length < 4500:
				short_counter += 1
				row = [time_begin, read.number, channel, read.id, signal_length, 'short']
				sampling_writer.writerow(row)
			else:
				read_list.append((channel, read, signal_length))
				normal_data = modified_zscore(raw_data[-3000:])
				inputs.append(normal_data)

		# input the model to determine whether it is the target sequence
		if len(inputs) > 0:
			inputs = np.array(inputs).to(device)
			outputs = model(inputs).max(dim=1).indices
			target_reads, non_target_reads = [], []
			for index, (channel, read, signal_length) in enumerate(read_list):
				if outputs[index] == 1:
					target_counter += 1
					target_reads.append((channel, read.number))
					row = [time_begin, read.number, channel, read.id, signal_length, 'stop_receiving']
					sampling_writer.writerow(row)
				else:
					non_target_counter += 1
					non_target_reads.append((channel, read.number))
					row = [time_begin, read.number, channel, read.id, signal_length, 'unblock']
					sampling_writer.writerow(row)
			client.stop_receiving_batch(target_reads)
			client.unblock_read_batch(non_target_reads, duration=unblock_duration)
		# limit the rate at which we make requests
		time_end = time.time()
		if time_begin + throttle > time_end:
			time.sleep(throttle + time_begin - time_end)
		if len(read_batch) > 0:
			print("batch time: {}, batch size: {}, target reads: {}, non-target reads: {}, short reads: {}, control group reads: {}".format(
				time_end-time_begin, len(read_batch), target_counter, non_target_counter, short_counter, control_counter))
	return target_counter


def run_workflow(
	client: read_until.ReadUntilClient,
	analysis_worker: typing.Callable[[], None],
	n_workers: int,
	run_time: float,
	runner_kwargs: typing.Optional[typing.Dict] = None,
):
	"""Run an analysis function against a ReadUntilClient.

	:param client: `ReadUntilClient` instance.
	:param analysis worker: a function to process reads. It should exit in
		response to `client.is_running == False`.
	:param n_workers: number of incarnations of `analysis_worker` to run.
	:param run_time: time (in seconds) to run workflow.
	:param runner_kwargs: keyword arguments for `client.run()`.

	:returns: a list of results, on item per worker.

	"""
	logger = logging.getLogger("Manager")

	if not runner_kwargs:
		runner_kwargs = {}

	results = []
	pool = ThreadPool(n_workers)
	logger.info("Creating %s workers", n_workers)
	try:
		# start the client
		client.run(**runner_kwargs)

		# start a pool of workers
		for _ in range(n_workers):
			results.append(pool.apply_async(analysis_worker))
		pool.close()

		# wait a bit before closing down
		time.sleep(run_time)
		logger.info("Sending reset")
		client.reset()
		pool.join()
	except KeyboardInterrupt:
		logger.info("Caught ctrl-c, terminating workflow.")
		client.reset()

	# collect results (if any)
	collected = []
	for result in results:
		try:
			res = result.get(3)
		except TimeoutError:
			logger.warning("Worker function did not exit successfully.")
			collected.append(None)
		except Exception:  # pylint: disable=broad-except
			logger.exception("Worker raise exception:")
		else:
			logger.info("Worker exited successfully.")
			collected.append(res)
	pool.terminate()
	return collected


def main(argv=None):
	"""simple example main cli entrypoint"""
	args = get_parser().parse_args(argv)

	logging.basicConfig(
		format="[%(asctime)s - %(name)s] %(message)s",
		datefmt="%H:%M:%S",
		level=args.log_level,
	)

	model = ReadCurrent([32, 64, 128, 256, 512], n_fc_neurons=2048, depth=29, shortcut=True)
	if args.gpu_ids:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = nn.DataParallel(model).to(device)
	model.load_state_dict(torch.load(args.model_state))
	
	print(f"Run in {device} {args.gpu_ids}")

	channel_credentials = None
	if args.ca_cert is not None:
		channel_credentials = grpc.ssl_channel_credentials(
			root_certificates=args.ca_cert.read_bytes()
		)
	read_until_client = read_until.ReadUntilClient(
		mk_host=args.host,
		mk_port=args.port,
		mk_credentials=channel_credentials,
		one_chunk=args.one_chunk,
		filter_strands=True,
	)

	analysis_worker = functools.partial(
		simple_analysis,
		model,
		device,
		client=read_until_client,
		output=args.output,
		delay=args.analysis_delay,
		unblock_duration=args.unblock_duration,
	)

	results = run_workflow(
		read_until_client,
		analysis_worker,
		args.workers,
		args.run_time,
		runner_kwargs={"min_chunk_size": args.min_chunk_size},
	)

	for idx, result in enumerate(results):
		logging.info("Worker %s received %s target reads", idx + 1, result)


if __name__ == '__main__':
	main()