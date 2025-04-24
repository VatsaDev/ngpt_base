import os
import math
import time
import glob
import torch
import string
import random
import pickle
import numpy as np
import itertools # Added for cycling through data shards
from pathlib import Path # Added for new data loader

import tiktoken # Added for GPT-2 tokenizer

import matplotlib
from matplotlib import pyplot as plt

import torch.cuda.amp as amp  # For GradScaler

# Assuming model.py exists and contains the Transformer class and its config dict
import model
from model import Transformer

# --- hyperparams ---

batch_size = 64
block_size = 128 # ctx_len
eval_interval = 20
grad_accum_steps = 4  # Num microbatches

lr = 1e-3
min_lr = 1e-4

max_iters = 200
eval_iters = 20
warmup_iters = 10 # Note: CosineAnnealingLR doesn't use warmup_iters directly in this setup, but kept for potential future use or reference

train_losses_history = []
val_losses_history = []

beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1

max_grad_norm = 1.0  # Grad clipping threshold

# --- continue or scratch ---

resume = False
resume_checkpoint = "checkpoints/iM3C8i_check_100.pt" # Example checkpoint

data_dir = "shakespeare" # Assumes data prepared in the new format is here

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.config['device'] = device # model.config likely doesn't need device, model is moved later

scaler = amp.GradScaler(enabled=(device == 'cuda'))  # GradScaler for mixed precision training, disable on CPU

dtype = torch.float32 # Default dtype
if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[str(dtype)]
ctx = torch.amp.autocast(device_type=device, dtype=ptdtype, enabled=(device=='cuda'))
print(f"Using device: {device}, dtype: {dtype}")

# --- Matplotlib Styling ---

plt.style.use('default') # Reset to default
matplotlib.rcParams['font.family'] = 'sans-serif' # Set global font to sans-serif
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'] # Ensure sans-serif fonts are prioritized
matplotlib.rcParams['axes.spines.top'] = False # Remove top spine
matplotlib.rcParams['axes.spines.right'] = False # Remove right spine
matplotlib.rcParams['axes.facecolor'] = '#f0f0f0' # Light background color for plot area
matplotlib.rcParams['figure.facecolor'] = '#f0f0f0' # Light background color for the figure
matplotlib.rcParams['grid.alpha'] = 0.4 # Make grid lines more subtle if you choose to use them later
matplotlib.rcParams['axes.titlesize'] = 12 # Reduced title fontsize
matplotlib.rcParams['axes.labelsize'] = 12 # Adjust axis label fontsize
matplotlib.rcParams['xtick.labelsize'] = 10 # Adjust x tick label fontsize
matplotlib.rcParams['ytick.labelsize'] = 10 # Adjust y tick label fontsize
matplotlib.rcParams['legend.fontsize'] = 10 # Adjust legend fontsize
matplotlib.rcParams['axes.titlecolor'] = 'grey' # Grey title color
matplotlib.rcParams['axes.labelcolor'] = 'grey' # Grey axis label color
matplotlib.rcParams['xtick.color'] = 'grey' # Grey x tick color
matplotlib.rcParams['ytick.color'] = 'grey' # Grey y tick color
matplotlib.rcParams['legend.labelcolor'] = 'grey' # Grey legend text color

# --- Run Name ---
characters = string.ascii_letters + string.digits  # Includes uppercase, lowercase letters, and digits
run_name = ''.join(random.choice(characters) for i in range(6))
if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
if not os.path.exists('plots'): os.makedirs('plots')

# --- Tiktoken Encoder ---
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
vocab_size = enc.n_vocab
print(f"Using GPT-2 tokenizer, vocab_size = {vocab_size}")

# --- Data Loading ---

# Helper function to load a single shard
def _load_data_shard(file: Path):
    # Use torch.from_file to read the header without numpy
    header = torch.from_file(str(file), shared=False, size=256, dtype=torch.int32)
    # It seems torch.from_file doesn't support offset directly for reading parts of the file after header easily.
    # Let's stick to the file opening approach but make sure it reads the header first.
    with file.open("rb") as f:
        # Read header bytes
        header_bytes = f.read(256 * 4)
        # Interpret header bytes as int32 tensor
        header = torch.frombuffer(header_bytes, dtype=torch.int32)

        # Validate header
        assert header[0].item() == 20240520, f"magic number mismatch in {file}, got {header[0].item()}"
        assert header[1].item() == 1, f"unsupported version in {file}, got {header[1].item()}"
        num_tokens = int(header[2].item()) # number of tokens (claimed)

        # Read token data directly into a tensor's memory
        # Create tensor first. Pin memory if using CUDA.
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=(device=='cuda'))
        # Get the underlying numpy array view IF NEEDED for readinto, but let's try to avoid
        # Direct read into tensor storage is harder. Let's use the numpy view approach from the example.
        # Ensure tensor is contiguous for numpy view
        tokens_np = tokens.numpy() # This shares memory

        # Read data into the numpy array view
        nbytes_read = f.readinto(tokens_np.data) # Use .data to get the buffer protocol object

        # Verify read bytes
        expected_bytes = 2 * num_tokens
        assert nbytes_read == expected_bytes, f"number of bytes read ({nbytes_read}) does not match header ({expected_bytes}) in {file}"

    # Data is now in the `tokens` tensor, potentially pinned
    return tokens


def create_data_generator(filename_pattern: str, batch_size: int, block_size: int, rank : int = 0, world_size : int = 1):
    """ Creates a generator for data loading. rank/world_size added for compatibility, assumes 0/1 """
    files = sorted(glob.glob(filename_pattern))
    if not files:
        raise FileNotFoundError(f"No data files found matching pattern: {filename_pattern}")
    print(f"Found {len(files)} data shards for pattern {filename_pattern}")

    # Use itertools.cycle to repeat shards indefinitely if needed
    file_iter = itertools.cycle([Path(file) for file in files])

    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    assert local_batch_size * world_size == batch_size # Ensure division is exact

    current_tokens = None
    current_pos = 0

    while True:
        # Load next shard if needed
        if current_tokens is None or current_pos + block_size * local_batch_size * world_size + 1 > len(current_tokens):
            next_file = next(file_iter)
            print(f"Loading next data shard: {next_file}")
            current_tokens = _load_data_shard(next_file)
            # Convert to target dtype (int64 for embedding indices) right after loading
            # Keep on CPU for now, transfer happens below.
            current_tokens = current_tokens.to(torch.int64)
            current_pos = 0
            if len(current_tokens) <= block_size + 1:
                 print(f"Warning: Shard {next_file} is smaller than block_size+1 ({len(current_tokens)} tokens). Skipping.")
                 current_tokens = None # Force loading next shard
                 continue # Skip to next iteration to load another shard


        # Determine the starting position for this rank's batch
        # In non-distributed (world_size=1), rank_start_offset is always 0
        rank_start_offset = current_pos + rank * local_batch_size * block_size

        # Prepare batch lists
        batch_x, batch_y = [], []
        for i in range(local_batch_size):
             # Calculate start and end indices for the i-th sequence in the local batch
             start_idx = rank_start_offset + i * block_size
             end_idx = start_idx + block_size

             # Check boundary conditions carefully
             if end_idx + 1 > len(current_tokens):
                 # This condition should ideally be caught by the shard loading logic earlier,
                 # but double-check to prevent index out of bounds.
                 print(f"Warning: Reached end of shard {next_file} unexpectedly. Resetting pos and trying again.")
                 current_tokens = None # Force reload
                 break # Break inner loop, outer loop will reload

             x = current_tokens[start_idx : end_idx]
             y = current_tokens[start_idx + 1 : end_idx + 1]
             batch_x.append(x)
             batch_y.append(y)

        if current_tokens is None: # If we had to break the inner loop and force reload
            continue # Skip to next iteration of the while loop

        # Stack sequences into batch tensors
        inputs = torch.stack(batch_x)
        targets = torch.stack(batch_y)

        # Move data to the target device
        inputs = inputs.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        # Update the global position across all ranks
        current_pos += batch_size * block_size

        yield inputs, targets


# Create data generators for train and val
train_data_pattern = os.path.join("data", data_dir, f"{data_dir}_train_*.bin")
val_data_pattern = os.path.join("data", data_dir, f"{data_dir}_val_*.bin")

# Note: The provided distributed_data_generator yields one batch element at a time.
# The original get_batch yielded a full batch. Let's adapt the generator
# concept to yield full batches. We'll modify the logic inside the generator.
# Let's rename the provided generator and create a wrapper or modify it.
# The structure seems more like yielding individual sequences than full batches?
# Rereading the provided `distributed_data_generator`: it slices `tokens`
# based on `rank` and `local_batch_size` but seems to yield single sequences `buf[:-1]` and `buf[1:]`.
# This seems incorrect for standard batch training.
# Let's REIMPLEMENT `get_batch` using the shard loading logic but constructing batches like the original script.

def get_batch_from_shards(split, data_gens):
    """ Gets a batch using the generator for the specified split """
    if split == 'train':
        X, Y = next(data_gens['train'])
    else: # split == 'val'
        X, Y = next(data_gens['val'])
    return X, Y

# Instantiate generators *outside* the loop
# Using rank 0, world_size 1 for single process execution
try:
    train_data_gen = create_data_generator(train_data_pattern, batch_size, block_size, rank=0, world_size=1)
    val_data_gen = create_data_generator(val_data_pattern, batch_size, block_size, rank=0, world_size=1)
    data_gens = {'train': train_data_gen, 'val': val_data_gen}
    print("Data generators created successfully.")
except FileNotFoundError as e:
    print(f"Error creating data generators: {e}")
    print("Please ensure that data files exist in the format required by _load_data_shard:")
    print(" - Files named like 'data/shakespeare/shakespeare_train_*.bin' and 'data/shakespeare/shakespeare_val_*.bin'")
    print(" - Each file has a header (256*4 bytes): magic_num (20240520), version (1), num_tokens")
    print(" - Followed by num_tokens * 2 bytes of uint16 token data.")
    exit(1)


# --- Model Init ---

# Update model config with vocab size *before* initializing
model.config["vocab_size"] = vocab_size
model.config["block_size"] = block_size # Ensure model knows block_size if needed

if resume:
    print(f"Resuming from checkpoint: {resume_checkpoint}")
    checkpoint = torch.load(resume_checkpoint, map_location=device)
    # TODO: Load model config from checkpoint if necessary, for now assume config matches
    # model_args = checkpoint['model_args'] # Example if config/args were saved
    model_instance = Transformer() # Initialize using updated global config

    state_dict = checkpoint['model']
    # Fix potential state dict keys mismatch if using compile/DDP etc.
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model_instance.load_state_dict(state_dict)
    m = model_instance.to(device)

    optimizer = m.configure_optimizers(weight_decay, lr, (beta1, beta2), device_type=device) # Pass device_type if needed by optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=min_lr) # Recreate scheduler

    # Restore previous run state
    start_iter = checkpoint['iter'] + 1 # Start from the next iteration
    run_name = checkpoint['run_name']
    try:
        train_losses_history = checkpoint['train_losses_history']
        val_losses_history = checkpoint['val_losses_history']
    except KeyError:
        print("Warning: Loss history not found in checkpoint. Starting fresh history.")
        train_losses_history = []
        val_losses_history = []

    print(f"Resumed from run {run_name} at iteration {start_iter}")

else:
    model_instance = Transformer() # Initialize using updated global config
    m = model_instance.to(device)
    optimizer = m.configure_optimizers(weight_decay, lr, (beta1, beta2), device_type=device) # Pass device_type if needed by optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=min_lr)
    start_iter = 0  # Start from iteration 0
    train_losses_history = [] # Initialize fresh history
    val_losses_history = []
    print(f"Starting new run {run_name} from scratch")

p = sum(p.numel() for p in m.parameters() if p.requires_grad) # Count trainable params
print(f"{p/1e6:.2f} M parameters")

# --- Compile the model ---
print("Compiling the model...")
# compiled_model = torch.compile(m) # Enable compilation
compiled_model = m # Keep uncompiled for simplicity first / debugging
print("Model compilation complete (or skipped).")


# --- Loss Estimation and Generation ---

@torch.no_grad()
def estimate_loss(model_to_eval, data_gens):
    out = {}
    model_to_eval.eval() # Set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_from_shards(split, data_gens)
            # No need for autocast context manager if estimate_loss is @no_grad
            # and model forward doesn't require it internally for inference
            logits, loss = model_to_eval(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # model_to_eval.train() # Set back to train mode AFTER generation below
    return out

@torch.no_grad()
def generate_text(model_to_gen, enc, max_new_tokens=100, temperature=0.8, top_k=200):
    """ Generates text using the model. Uses the *original* model for generation. """
    model_to_gen.eval() # Ensure eval mode
    # Start with a newline character prompt
    start = "\n"
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # Run generation
    print("Generating text...")
    with ctx: # Use autocast for generation if beneficial
        y = model_to_gen.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    generated_ids = y[0].tolist() # Get generated sequence for the first (and only) batch item
    generated_text = decode(generated_ids)

    print("--- Generated Text Start ---")
    print(generated_text)
    print("--- Generated Text End ---")
    model_to_gen.train() # Set back to train mode


# --- Training Loop ---

time_s = time.time()
prev_time = time_s  # track previous step time

# Ensure optimizer state is on the correct device, especially after loading checkpoint
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

optimizer.zero_grad(set_to_none=True) # Initialize gradients to None

for iter_num in range(start_iter, max_iters + 1):

    # determine and set the learning rate for this iteration
    # Implements linear warmup and cosine decay schedule
    lr_iter = min_lr # Default to min_lr
    if iter_num < warmup_iters:
        # Linear warmup
        lr_iter = lr * iter_num / warmup_iters
    elif iter_num <= max_iters:
        # Cosine decay
        decay_ratio = (iter_num - warmup_iters) / (max_iters - warmup_iters)
        assert 0.0 <= decay_ratio <= 1.0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        lr_iter = min_lr + coeff * (lr - min_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_iter
    current_lr = lr_iter # For printing

    # --- Evaluation Phase ---
    if iter_num % eval_interval == 0 or iter_num == max_iters:
        # Important: Use the *original* model 'm' for estimate_loss and generation
        # as compiled models might behave differently or hide attributes.
        losses = estimate_loss(m, data_gens)
        val_loss = losses['val']
        val_losses_history.append(val_loss) # Append validation loss

        time_n = time.time()
        elapsed = time_n - time_s
        dt = time_n - prev_time # time since last eval step
        prev_time = time_n

        # MFU calculation - use original model 'm' and its config
        mfu = m.estimate_mfu(p, batch_size * grad_accum_steps, dt) if hasattr(m, 'estimate_mfu') else 0.0

        print(f"step: {iter_num}, train loss: {losses['train']:.4f}, val loss: {val_loss:.4f}, lr: {current_lr:.6f}, elapsed: {elapsed/60:.2f} min, MFU: {mfu*100:.2f}%")

        # --- Generate Text ---
        # Use the original model 'm' for generation
        if hasattr(m, 'generate'):
             generate_text(m, enc, max_new_tokens=100) # Generate 100 tokens
        else:
             print("Warning: model does not have a .generate() method. Skipping text generation.")

        # --- Checkpointing ---
        if iter_num > 0: # Don't save checkpoint at step 0 if not resuming
            checkpoint = {
                'model': m.state_dict(), # Save original model state dict
                'optimizer': optimizer.state_dict(),
                'iter': iter_num,
                'run_name': run_name,
                'config': model.config, # Save config used for this run
                'train_losses_history': train_losses_history,
                'val_losses_history': val_losses_history,
            }
            ckpt_path = f'checkpoints/{run_name}_check_{iter_num}.pt'
            print(f"Saving checkpoint to {ckpt_path}")
            torch.save(checkpoint, ckpt_path)

        # --- Plotting ---
        plt.figure(figsize=(8, 4), dpi=100)
        ax = plt.gca()

        # X-axis for validation losses (every eval_interval steps)
        iterations_eval = range(0, iter_num + 1, eval_interval)
        # X-axis for training losses (every step)
        iterations_train = range(iter_num) # Up to current iter_num (exclusive because we log loss *after* step)

        # Ensure plot data lengths match x-axis lengths
        train_losses_to_plot = train_losses_history[:len(iterations_train)]
        val_losses_to_plot = val_losses_history[:len(iterations_eval)] # val history grows every eval_interval

        if len(train_losses_to_plot) > 0 :
            train_line = plt.plot(iterations_train, train_losses_to_plot, label='Train', color='royalblue', linestyle='-', linewidth=1.5, marker='', alpha=0.8)
            plt.plot(iterations_train[-1:], train_losses_to_plot[-1:], marker='o', markersize=3, markerfacecolor='royalblue', markeredgecolor='none', linestyle='none')

        if len(val_losses_to_plot) > 0:
            val_line = plt.plot(iterations_eval, val_losses_to_plot, label='Val', color='palevioletred', linestyle='-', linewidth=1.5, marker='', alpha=0.8)
            plt.plot(iterations_eval[-1:], val_losses_to_plot[-1:], marker='o', markersize=3, markerfacecolor='palevioletred', markeredgecolor='none', linestyle='none')


        plt.xlabel("Steps", labelpad=8, color='grey')
        plt.ylabel("Loss", labelpad=8, color='grey')
        plt.title(f"Train/Val Loss - Run: {run_name}", fontsize=12, color='grey')
        legend = plt.legend(frameon=False, loc='upper right')
        for line in legend.get_lines():
            line.set_linewidth(2.0)
            line.set_solid_capstyle('round')

        ax.tick_params(axis='both', which='major', pad=8)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # Dynamic Y-axis Ticks (optional refinement)
        all_losses = train_losses_history + val_losses_history
        if all_losses:
            min_loss = min(all_losses) if all_losses else 0
            max_loss = max(all_losses) if all_losses else 1
            # Example: Set ticks based on range, adjust step as needed
            y_ticks = np.linspace(min_loss, max_loss, num=5) # Adjust num for density
            # ax.set_yticks(y_ticks)

        plt.grid(axis='y', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

        plot_path = f"plots/{run_name}_plot_{iter_num}.png"
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        print(f"Plot saved to {plot_path}")
        plt.close() # Close figure to free memory

    # --- Training Step ---
    if iter_num == max_iters: break # Exit after final evaluation

    # Set model to train mode *before* the training step
    m.train()
    compiled_model.train() # Ensure compiled model is also in train mode if used

    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        xb, yb = get_batch_from_shards('train', data_gens)

        # Forward pass using compiled model and autocast
        with ctx:
            # Use compiled_model here for potential speedup
            logits, loss = compiled_model(xb, yb)
            loss = loss / grad_accum_steps # Scale loss for gradient accumulation

        # Backward pass with scaler
        # Accumulate scaled gradients
        scaler.scale(loss).backward()
        loss_accum += loss.item() * grad_accum_steps # Unscale loss for logging

    # Record average loss for the entire step *before* optimizer step
    train_losses_history.append(loss_accum)

    # Gradient Clipping (unscale first)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(m.parameters(), max_grad_norm)

    # Optimizer Step
    scaler.step(optimizer)
    scaler.update()

    # Flush gradients
    optimizer.zero_grad(set_to_none=True)

    # scheduler.step() # Scheduler step was moved to evaluation block in original? No, CosineAnnealingLR steps every iter. Let's keep it here.
    # Correction: LR calculation is now manual based on iter_num, so scheduler.step() is not strictly needed
    # if we manually set LR each step. However, standard practice is to call scheduler.step(). Let's check if
    # CosineAnnealingLR *requires* step() to update internal state. Yes, it does. Keep scheduler.step().
    # Let's remove the manual LR calculation and rely solely on the scheduler + warmup logic if needed.
    # Reverting to standard scheduler usage:
    # Remove manual LR calculation block above.
    # Call scheduler.step() here. Add warmup logic separately if needed.

    # Let's keep the manual LR calculation for now as it explicitly includes warmup,
    # and CosineAnnealingLR itself doesn't handle warmup directly this way.
    # So, we set optimizer LR manually and *don't* call scheduler.step().


# --- End of Training ---
print('Training finished.')

# Final save? (Optional, could rely on last checkpoint)
final_ckpt_path = f'checkpoints/{run_name}_final.pt'
print(f"Saving final model to {final_ckpt_path}")
final_checkpoint = {
    'model': m.state_dict(),
    'optimizer': optimizer.state_dict(),
    'iter': max_iters,
    'run_name': run_name,
    'config': model.config,
    'train_losses_history': train_losses_history,
    'val_losses_history': val_losses_history,
}
torch.save(final_checkpoint, final_ckpt_path)

# Final plot
# (Plotting code is already inside the loop, last plot is generated at max_iters)
print(f"Final plot saved to plots/{run_name}_plot_{max_iters}.png")
