import os
import pickle
import numpy as np
import torch
import glob
from tqdm import tqdm
import multiprocessing as mp

# --- hyperparameters ---
shard_size = 1000000 # 1 million tokens per shard
data_dir = "data" # parent folder for all datasets
block_size = 256 # context length of the model
batch_size = 64 # batch size for data loading
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# --- --- ---

# Get dataset folder from user input
dataset_folder = "shakespeare"
dataset_path = dataset_folder
DATA_CACHE_DIR = dataset_path # reuse variable name for output

# Check if the dataset folder exists
if not os.path.exists(dataset_path):
    print(f"Error: Dataset folder not found at {dataset_path}")
    exit()

# Read all text files in the dataset folder
input_files = glob.glob(os.path.join(dataset_path, "*.txt"))
if not input_files:
    print(f"Error: No .txt files found in {dataset_path}")
    exit()

data = ""
for file_path in input_files:
    with open(file_path, 'r') as f:
        data += f.read()

chars = sorted(list(set(data)))
vocab_size = len(chars)

# token mappings
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]
def decode(l):
    return ''.join([itos[i] for i in l])

# tokenize the whole dataset
all_ids = encode(data)
all_ids_np = np.array(all_ids, dtype=np.uint16) # tokenize and convert to numpy array once

# train/val split (before sharding for consistent split across shards)
n = len(all_ids_np)
train_ids = all_ids_np[:int(n*0.9)]
val_ids = all_ids_np[int(n*0.9):]

# --- Sharding logic ---
def write_datafile(filename, data_shard):
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic number for cross-implementation compatibility
    header[1] = 1 # data format version
    header[2] = len(data_shard)
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(data_shard.tobytes())

def shard_dataset(ids, split_name):
    shard_index = 0
    token_count = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16) # preallocate buffer
    progress_bar = tqdm(total=len(ids), unit="tokens", desc=f"Sharding {split_name}")

    for token in ids:
        if token_count < shard_size:
            all_tokens_np[token_count] = token
            token_count += 1
        else:
            # write current shard and start a new one
            filename = os.path.join(DATA_CACHE_DIR, f"{dataset_folder}_{split_name}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            token_count = 0
            all_tokens_np = np.empty((shard_size,), dtype=np.uint16) # re-preallocate
            all_tokens_np[token_count] = token # put current token in new shard
            token_count += 1
        progress_bar.update(1)

    # write last shard if any tokens remaining
    if token_count > 0:
        filename = os.path.join(DATA_CACHE_DIR, f"{dataset_folder}_{split_name}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count]) # only write filled part
    progress_bar.close()
    print(f"Sharded {split_name} data into {shard_index + (1 if token_count > 0 else 0)} files.")


print("Sharding training data...")
shard_dataset(train_ids, "train")
print("Sharding validation data...")
shard_dataset(val_ids, "val")


# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'block_size': block_size, # also save block_size and other relevant params for later use
    'vocab_source': 'char' # or 'bpe' or 'word' etc
}

meta_pkl_path = os.path.join(DATA_CACHE_DIR, 'meta.pkl')
with open(meta_pkl_path, 'wb') as f:
    pickle.dump(meta, f)

print(f"Dataset files and meta.pkl created in: {DATA_CACHE_DIR}")

# Add dataset folder to gitignore
gitignore_path = '../../.gitignore'
dataset_gitignore_entry = f"{data_dir}/{dataset_folder}/\n"

if os.path.exists(gitignore_path):
    with open(gitignore_path, 'r') as f:
        gitignore_content = f.readlines()

    found_dataset_entry = False
    for line in gitignore_content:
        if line.strip() == dataset_gitignore_entry.strip():
            found_dataset_entry = True
            break

    if not found_dataset_entry:
        with open(gitignore_path, 'a') as f:
            f.write(dataset_gitignore_entry)
            print(f"Added '{dataset_gitignore_entry.strip()}' to .gitignore")
    else:
        print(f"'{dataset_gitignore_entry.strip()}' already in .gitignore")

else:
    with open(gitignore_path, 'w') as f:
        f.write(dataset_gitignore_entry)
        print(f".gitignore created and added '{dataset_gitignore_entry.strip()}'")
