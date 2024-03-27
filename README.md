# ReadCurrent: A VDCNN based tool for fast and accurate nanopore selective sequencing


## Initialization with Conda
### 1. Create a virtual environment by conda
```
conda create -n ReadCurrent python==3.9
```

### 2. Install PyTorch
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

### 3. Install other required libraries
```
pip install -r requirements.txt
```


## Scripts
### get_ids
Get the ids of reads that were successfully aligned (mapping quality >= 10) to the reference genome

```
config arguments:
  fastq_path            The directory where the fastq files is located, all fastq/fastq.gz files should be in the same folder
  ref_path              The path of the reference file
  align_threads         Number of threads using minimap2 for sequence alignment
  output                Output path for alignment results and read ids
```

Example:
```
snakemake -s tools/get_ids.smk --config fastq_path={fastq_path} ref_path={ref_path} align_threads=16 output={output} --cores 1
```

### read_fast5
Constructing training, validation, and testing sets from the fast5 files of nanopore sequencing data

```
usage: read_fast5.py [-h] --file_dir FILE_DIR --output OUTPUT [--read_ids READ_IDS] [--min_length MIN_LENGTH] [--train_size TRAIN_SIZE] [--valid_size VALID_SIZE] [--test_size TEST_SIZE]

Read fast5

optional arguments:
  -h, --help            show this help message and exit
  --file_dir FILE_DIR, -dir FILE_DIR
                        The directory where the fast5 files is located
  --output OUTPUT, -o OUTPUT
                        Storage path for output files
  --read_ids READ_IDS, -ids READ_IDS
                        The path for read ids file
  --min_length MIN_LENGTH, -len MIN_LENGTH
                        Minimum length of each electrical signal, default 4500
  --train_size TRAIN_SIZE, -train TRAIN_SIZE
                        Number of electrical signals to be read for training, default 20000
  --valid_size VALID_SIZE, -valid VALID_SIZE
                        Number of electrical signals to be read for validation, default 10000
  --test_size TEST_SIZE, -test TEST_SIZE
                        Number of electrical signals to be read for testing, default 10000
```

Example:
```
python tools/read_fast5.py -dir {fast5_dir} -o {output} -ids {read_ids_path}
```

### preprocessor
Perform data preprocessing on training and validation sets from the dataset folder

```
usage: preprocessor.py [-h] --data_folder DATA_FOLDER [--cut CUT] [--tiling_fold TILING_FOLD] [--length LENGTH] [--patches] [--seq_length SEQ_LENGTH] [--stride STRIDE] [--patch_size PATCH_SIZE]

Data preprocessing

optional arguments:
  -h, --help            show this help message and exit
  --data_folder DATA_FOLDER, -d DATA_FOLDER
                        Path to the dataset folder that contains train, valid, test files (.npy)
  --cut CUT, -c CUT     Electrical signal length to be cut, default 1500
  --tiling_fold TILING_FOLD, -tf TILING_FOLD
                        Number of tiles, default 3
  --length LENGTH, -l LENGTH
                        The length of the sliding window, default 3000
  --patches, -patches   Convert electrical signals into patches, default False
  --seq_length SEQ_LENGTH, -sl SEQ_LENGTH
                        Sequence length after patch, default 299
  --stride STRIDE, -s STRIDE
                        Patch step size, default 10
  --patch_size PATCH_SIZE, -ps PATCH_SIZE
                        The size of patch, default 16
```

Example:
```
python preprocessor.py -d {dataset_folder}
```

### trainer
Train the model on the specified dataset

```
usage: trainer.py [-h] --pos_data_folder POS_DATA_FOLDER --neg_data_folder NEG_DATA_FOLDER --output OUTPUT [--preprocess] [--cut CUT] [--tiling_fold TILING_FOLD] [--length LENGTH] [--patches]
                  [--seq_length SEQ_LENGTH] [--stride STRIDE] [--patch_size PATCH_SIZE] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--learning_rate LEARNING_RATE] [--tolerance TOLERANCE] [--interm INTERM]
                  [--num_workers NUM_WORKERS] [--gpu_ids GPU_IDS]

Training model

optional arguments:
  -h, --help            show this help message and exit
  --pos_data_folder POS_DATA_FOLDER, -p POS_DATA_FOLDER
                        Path to the positive dataset folder that contains train, valid, test files (.npy)
  --neg_data_folder NEG_DATA_FOLDER, -n NEG_DATA_FOLDER
                        Path to the negative dataset folder that contains train, valid, test files (.npy)
  --output OUTPUT, -o OUTPUT
                        The output path
  --preprocess, -preprocess
                        Whether to preprocess the training and validation dataset, default False
  --cut CUT, -c CUT     Electrical signal length to be cut, default 1500
  --tiling_fold TILING_FOLD, -tf TILING_FOLD
                        Number of tiles, default 3
  --length LENGTH, -l LENGTH
                        The length of the sliding window, default 3000
  --patches, -patches   Convert electrical signals into patches, default False
  --seq_length SEQ_LENGTH, -sl SEQ_LENGTH
                        Sequence length after patch, default 299
  --stride STRIDE, -s STRIDE
                        Patch step size, default 10
  --patch_size PATCH_SIZE, -ps PATCH_SIZE
                        The size of patch, default 16
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size, default 1024
  --epochs EPOCHS, -e EPOCHS
                        Number of epoches, default 300
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        Learning rate, default 1e-3
  --tolerance TOLERANCE, -t TOLERANCE
                        Tolerance for non increase in accuracy during training, default 10
  --interm INTERM, -i INTERM
                        The path for model checkpoint, default None
  --num_workers NUM_WORKERS, -nw NUM_WORKERS
                        The size of num_workers in Dataloader, default 0
  --gpu_ids GPU_IDS, -g GPU_IDS
                        Specify the GPU to use, if not specified, use all GPUs or CPU, default None
```

Example:
```
python trainer.py -p {pos_data_folder} -n {neg_data_folder} -o {output} -g 0
```

### tester
Test the model on the specified dataset

```
usage: tester.py [-h] --pos_data_folder POS_DATA_FOLDER --neg_data_folder NEG_DATA_FOLDER --model_state MODEL_STATE --output OUTPUT [--batch_size BATCH_SIZE] [--cut CUT] [--length LENGTH] [--patches]
                 [--seq_length SEQ_LENGTH] [--stride STRIDE] [--patch_size PATCH_SIZE] [--gpu_ids GPU_IDS]

Test model

optional arguments:
  -h, --help            show this help message and exit
  --pos_data_folder POS_DATA_FOLDER, -p POS_DATA_FOLDER
                        Path to the positive dataset folder that contains train, valid, test files (.npy)
  --neg_data_folder NEG_DATA_FOLDER, -n NEG_DATA_FOLDER
                        Path to the negative dataset folder that contains train, valid, test files (.npy)
  --model_state MODEL_STATE, -ms MODEL_STATE
                        Path of the model (a pth file)
  --output OUTPUT, -o OUTPUT
                        The output path
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size, default 512
  --cut CUT, -c CUT     Electrical signal length to be cut, default 1500
  --length LENGTH, -len LENGTH
                        The length of each signal segment, default 3000
  --patches, -patches   Convert electrical signals into patches, default False
  --seq_length SEQ_LENGTH, -sl SEQ_LENGTH
                        Sequence length after patch, default 299
  --stride STRIDE, -s STRIDE
                        Patch step size, default 10
  --patch_size PATCH_SIZE, -ps PATCH_SIZE
                        The size of patch, default 16
  --gpu_ids GPU_IDS, -g GPU_IDS
                        Specify the GPU to use, if not specified, use all GPUs or CPU, default None
```

Example:
```
python tester.py -p {pos_data_folder} -n {neg_data_folder} -ms {model_state} -o {output} -g 0
```