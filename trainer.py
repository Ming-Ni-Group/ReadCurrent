import os
import time
import argparse
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from sklearn.utils import shuffle

from dataset import Dataset
from preprocessor import train_normalization, valid_normalization
from tester import test
from models.ReadCurrent import ReadCurrent
from models.LSTM import LSTM
from models.CNN_LSTM import CNN_LSTM
from models.Transformer import Transformer
from models.CNN_Transformer import CNN_Transformer


def myprint(string, log):
	log.write(string+'\n')
	print(string)


def plot(train_acc, train_loss, valid_acc, valid_loss):
	plt.figure(figsize=(10, 5))
	plt.subplot(1, 2, 1)
	plt.plot(train_acc, label='train_acc')
	plt.plot(valid_acc, label='valid_acc')
	plt.xlabel('iter (x200)')
	plt.title('Accuracy')
	plt.legend()
	plt.subplot(1, 2, 2)
	plt.plot(train_loss, label='train_loss')
	plt.plot(valid_loss, label='valid_loss')
	plt.xlabel('iter (x200)')
	plt.title('Loss')
	plt.legend()
	plt.savefig(os.path.join(args.output, 'train.pdf'))
	plt.close()


def train(model, pos_train_generator, neg_train_generator, pos_valid_generator, neg_valid_generator, log, device):
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	scheduler = torch.optim.lr_scheduler.LambdaLR(
			optimizer, lr_lambda=lambda epoch: 1 / (2**epoch), last_epoch=-1, verbose=True)

	train_acc, valid_acc, train_loss, valid_loss = [], [], [], []
	train_acc_count, train_loss_count, best_acc, best_loss = 0, 0, 0, 1e7
	total_time, pos_epoch, neg_epoch = 0, 1, 1
	patience, scheduler_num, i = 0, 0, 0

	# Setting the tqdm progress bar
	train_iter = tqdm.tqdm(pos_train_generator,
							desc="training epoch %d" % (pos_epoch),
							total=len(pos_train_generator),
							bar_format="{l_bar}{r_bar}")
	neg_iter = neg_train_generator.__iter__()
	neg_val_iter = neg_valid_generator.__iter__()

	# Training
	start_time = time.time()
	while pos_epoch <= args.epochs:
		lr = optimizer.state_dict()['param_groups'][0]['lr']
		
		for pos_x, pos_y in train_iter:
			i += 1
			try:
				neg_x, neg_y = neg_iter.__next__()
			except:
				neg_epoch += 1
				neg_iter = neg_train_generator.__iter__()
				neg_x, neg_y = neg_iter.__next__()
			spx, spy = torch.cat((pos_x, neg_x)), torch.cat((pos_y, neg_y))
			spx, spy = shuffle(spx, spy)
			spx, spy = spx.type(torch.FloatTensor).to(device), spy.type(torch.LongTensor).to(device)

			# Forward pass
			outputs = model(spx)
			loss = criterion(outputs, spy)
			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# Calculate accuracy and loss
			acc = (spy == outputs.max(dim=1).indices).float().mean().item()
			l = loss.float().mean().item()
			train_acc_count += acc
			train_loss_count += l

			# Print information
			if i % 50 == 0:
				avg_train_acc, avg_train_loss = round(train_acc_count * 2, 2), round(train_loss_count / 50, 4)
				train_acc_count, train_loss_count = 0, 0
				post_fix = {
					"pos epoch": pos_epoch,
					"neg epoch": neg_epoch,
					"iter": i,
					"avg_loss": avg_train_loss,
					"avg_acc": avg_train_acc,
					"lr": lr
				}
				train_iter.write(str(post_fix))
				print(str(post_fix), file=log)

			# Validation
			if i % 200 == 0:
				train_acc.append(avg_train_acc)
				train_loss.append(avg_train_loss)

				model.eval()
				with torch.no_grad():
					valid_acc_count, valid_loss_count, vti = 0, 0, 0
					for pos_val_x, pos_val_y in pos_valid_generator:
						vti += 1
						try:
							neg_val_x, neg_val_y = neg_val_iter.__next__()
						except:
							neg_val_iter = neg_valid_generator.__iter__()
							neg_val_x, neg_val_y = neg_val_iter.__next__()

						valx, valy = torch.cat((pos_val_x, neg_val_x)), torch.cat((pos_val_y, neg_val_y))
						valx, valy = shuffle(valx, valy)
						valx, valy = valx.type(torch.FloatTensor).to(device), valy.type(torch.LongTensor).to(device)
						# Forward pass
						outputs_val = model(valx)
						# Calculate accuracy and loss
						loss_v = criterion(outputs_val, valy)

						valid_acc_count += (valy == outputs_val.max(dim=1).indices).float().mean().item()
						valid_loss_count += loss_v.float().mean().item()

					avg_valid_acc, avg_valid_loss = round((valid_acc_count / vti) * 100, 2), round(valid_loss_count / vti, 4)
					valid_acc.append(avg_valid_acc)
					valid_loss.append(avg_valid_loss)
					# Print information
					post_fix = {
						"valid_loss": avg_valid_loss,
						"valid_acc": avg_valid_acc,
						"best_loss": best_loss,
						"best_acc": best_acc,
					}
					train_iter.write(str(post_fix))
					print(str(post_fix), file=log)

					# Plot accuracy and loss
					plot(train_acc, train_loss, valid_acc, valid_loss)

					# Save model
					if best_loss <= avg_valid_loss:
						patience += 1
					else:
						patience = 0
						best_acc = avg_valid_acc
						best_loss = avg_valid_loss
						torch.save(model.state_dict(), os.path.join(args.output, "model.pth"))

					# Decay learning rate or end training early
					if patience >= args.tolerance and scheduler_num < 4:
						model.load_state_dict(torch.load(os.path.join(args.output, "model.pth")))
						scheduler.step()
						scheduler_num += 1
						patience = 0
						lr = optimizer.state_dict()['param_groups'][0]['lr']
					elif patience >= args.tolerance and scheduler_num >= 4:
						total_time = time.time() - start_time
						myprint("An early closure!", log)
						myprint("train acc: {}, train loss: {}, total time: {:.4f}, average epoch time: {:.4f}".format(
							best_acc, best_loss, total_time, (total_time / i) * len(pos_train_generator)), log)
						return
				model.train()
		pos_epoch += 1

	total_time = time.time() - start_time
	myprint("train acc: {}, train loss: {}, total time: {:.4f}, average epoch time: {:.4f}".format(
		best_acc, best_loss, total_time, (total_time / i) * len(pos_train_generator)), log)


if __name__ == '__main__':
	torch.manual_seed(3407)
	torch.set_num_threads(2)
	# Get command arguments
	parser = argparse.ArgumentParser(description="Training model")
	parser.add_argument("--pos_data_folder", '-p', type=str, required=True, help="Path to the positive dataset folder that contains train, valid, test files (.npy)")
	parser.add_argument("--neg_data_folder", '-n', type=str, required=True, help="Path to the negative dataset folder that contains train, valid, test files (.npy)")
	parser.add_argument("--output", '-o', type=str, required=True, help="The output path")
	parser.add_argument("--preprocess", '-preprocess', action='store_true', help="Whether to preprocess the training and validation dataset, default False")
	parser.add_argument("--cut", '-c', type=int, default=1500, help="Electrical signal length to be cut, default 1500")
	parser.add_argument("--tiling_fold", '-tf', type=int, default=3, help="Number of tiles, default 3")
	parser.add_argument("--length", '-l', type=int, default=3000, help="The length of the sliding window, default 3000")
	parser.add_argument("--patches", '-patches', action='store_true', help="Convert electrical signals into patches, default False")
	parser.add_argument("--seq_length", '-sl', type=int, default=299, help="Sequence length after patch, default 299")
	parser.add_argument("--stride", '-s', type=int, default=10, help="Patch step size, default 10")
	parser.add_argument("--patch_size", '-ps', type=int, default=16, help="The size of patch, default 16")
	parser.add_argument("--batch_size", '-b', type=int, default=1024, help="Batch size, default 1024")
	parser.add_argument("--epochs", '-e', type=int, default=300, help="Number of epoches, default 300")
	parser.add_argument("--learning_rate", '-lr', type=float, default=1e-3, help="Learning rate, default 1e-3")
	parser.add_argument("--tolerance", '-t', type=int, default=10, help="Tolerance for non increase in accuracy during training, default 10")
	parser.add_argument("--interm", '-i', type=str, default=None, help="The path for model checkpoint, default None")
	parser.add_argument("--num_workers", '-nw', type=int, default=0, help="The size of num_workers in Dataloader, default 0")
	parser.add_argument("--gpu_ids", '-g', type=str, default=None, help="Specify the GPU to use, if not specified, use all GPUs or CPU, default None")
	args = parser.parse_args()

	# Create output folder
	if not os.path.exists(args.output):
		os.makedirs(args.output)
	log = open(args.output+'/train.txt', mode='w', encoding='utf-8')

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
	myprint(f"Train in {device} {args.gpu_ids}", log)

	# Load model state
	if args.interm is not None:
		myprint(f"Load model from {args.interm}", log)
		model.load_state_dict(torch.load(args.interm))

	if args.preprocess:
		# load the training and validation set
		myprint("===================== Load the training and validation set =====================", log)
		pos_train_data = np.load(os.path.join(args.pos_data_folder, 'train.npy'), allow_pickle=True)
		myprint(f"Load the positive training set from {os.path.join(args.pos_data_folder, 'train.npy')}, shape {pos_train_data.shape}", log)
		neg_train_data = np.load(os.path.join(args.neg_data_folder, 'train.npy'), allow_pickle=True)
		myprint(f"Load the negative training set from {os.path.join(args.neg_data_folder, 'train.npy')}, shape {neg_train_data.shape}", log)

		pos_valid_data = np.load(os.path.join(args.pos_data_folder, 'valid.npy'), allow_pickle=True)
		myprint(f"Load the positive validation set from {os.path.join(args.pos_data_folder, 'valid.npy')}, shape {pos_valid_data.shape}", log)
		neg_valid_data = np.load(os.path.join(args.neg_data_folder, 'valid.npy'), allow_pickle=True)
		myprint(f"Load the negative validation set from {os.path.join(args.neg_data_folder, 'valid.npy')}, shape {neg_valid_data.shape}", log)
		myprint(f"===================== Load finished! =====================\n", log)

		# Preprocess the dataset
		myprint("===================== Start preprocessing! =====================", log)
		pos_train_data = train_normalization(pos_train_data, args.cut, args.length, args.tiling_fold,
									   args.patches, args.seq_length, args.stride, args.patch_size)
		myprint(f"The shape of the positive training set after preprocessing: {pos_train_data.shape}", log)
		neg_train_data = train_normalization(neg_train_data, args.cut, args.length, args.tiling_fold,
									   args.patches, args.seq_length, args.stride, args.patch_size)
		myprint(f"The shape of the negative training set after preprocessing: {neg_train_data.shape}", log)

		pos_valid_data = valid_normalization(pos_valid_data, args.cut, args.length,
									   args.patches, args.seq_length, args.stride, args.patch_size)
		myprint(f"The shape of the positive validation set after preprocessing: {pos_valid_data.shape}", log)
		neg_valid_data = valid_normalization(neg_valid_data, args.cut, args.length,
									   args.patches, args.seq_length, args.stride, args.patch_size)
		myprint(f"The shape of the negative validation set after preprocessing: {neg_valid_data.shape}", log)
		myprint("===================== Preprocessing finished! =====================\n", log)

	else:
		# load the preprocessed training and validation set
		myprint("===================== Load the preprocessed training and validation set =====================", log)
		pos_train_data = np.load(os.path.join(args.pos_data_folder, 'train_preprocessed.npy'), allow_pickle=True)
		myprint(f"Load the positive training set from {os.path.join(args.pos_data_folder, 'train_preprocessed.npy')}, shape {pos_train_data.shape}", log)
		neg_train_data = np.load(os.path.join(args.neg_data_folder, 'train_preprocessed.npy'), allow_pickle=True)
		myprint(f"Load the negative training set from {os.path.join(args.neg_data_folder, 'train_preprocessed.npy')}, shape {neg_train_data.shape}", log)

		pos_valid_data = np.load(os.path.join(args.pos_data_folder, 'valid_preprocessed.npy'), allow_pickle=True)
		myprint(f"Load the positive validation set from {os.path.join(args.pos_data_folder, 'valid_preprocessed.npy')}, shape {pos_valid_data.shape}", log)
		neg_valid_data = np.load(os.path.join(args.neg_data_folder, 'valid_preprocessed.npy'), allow_pickle=True)
		myprint(f"Load the negative validation set from {os.path.join(args.neg_data_folder, 'valid_preprocessed.npy')}, shape {neg_valid_data.shape}", log)
		myprint(f"===================== Load finished! =====================\n", log)

	# DataLoader parameters
	params = {'batch_size': args.batch_size // 2,
			  'shuffle': True,
			  'pin_memory': True,
			  'num_workers': args.num_workers}

	# Load files and create Dataset
	pos_train_set = Dataset(pos_train_data, 'pos')
	pos_train_generator = DataLoader(pos_train_set, **params)
	neg_train_set = Dataset(neg_train_data, 'neg')
	neg_train_generator = DataLoader(neg_train_set, **params)

	pos_valid_set = Dataset(pos_valid_data, 'pos')
	pos_valid_generator = DataLoader(pos_valid_set, **params)
	neg_valid_set = Dataset(neg_valid_data, 'neg')
	neg_valid_generator = DataLoader(neg_valid_set, **params)

	# Training
	myprint("===================== Start training! =====================", log)
	train(model, pos_train_generator, neg_train_generator, pos_valid_generator, neg_valid_generator, log, device)
	myprint("===================== Training finished! =====================\n", log)

	# Load the test set
	pos_test_data = np.load(os.path.join(args.pos_data_folder, 'test.npy'), allow_pickle=True)
	myprint(f'load the positive test set from {os.path.join(args.pos_data_folder, "test.npy")}, shape {pos_test_data.shape}', log)
	neg_test_data = np.load(os.path.join(args.neg_data_folder, 'test.npy'), allow_pickle=True)
	myprint(f'load the negative test set from {os.path.join(args.neg_data_folder, "test.npy")}, shape {neg_test_data.shape}', log)

	# Test
	myprint("===================== Start testing! =====================", log)
	model.load_state_dict(torch.load(os.path.join(args.output, "model.pth")))
	tp, fn, pos_infer_time = test(model, pos_test_data, 1, args.batch_size, args.cut, args.length,
			  args.patches, args.seq_length, args.stride, args.patch_size, log, device)
	tn, fp, neg_infer_time = test(model, neg_test_data, 0, args.batch_size, args.cut, args.length,
			  args.patches, args.seq_length, args.stride, args.patch_size, log, device)
	
	# Calculate evaluation index values
	accuracy = round((tp + tn) * 100 / (tp + tn + fp + fn), 2)
	precision = round(tp * 100 / (tp + fp), 2)
	recall = round(tp * 100 / (tp + fn), 2)
	f1_score = round((2 * precision * recall) / (precision + recall), 2)
	aver_infer_time = round((pos_infer_time + neg_infer_time) / 2, 4)
	myprint(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1_score: {f1_score}, average inference time: {aver_infer_time}", log)
	myprint("===================== Testing finished! =====================", log)
	log.close()