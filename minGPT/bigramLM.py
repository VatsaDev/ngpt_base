#import torch
import torch.nn as nn
from torch.nn import functional as F 

# hyperparams 

batch_size = 32
block_size = 8
eval_interval = 300
lr = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

# data file

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print('data read')

# tokenizer

print('tokenization')

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# tokenized train, val 

data = torch.tensor(encode(text))

n = int(0.9*len(data))

train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data)-block_size,(batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x, y

@torch.no_grad()
def estimate_loss():
    
    out = {}
    model.eval()

    for split in ['train','val']:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = losses.item()

        out[split] = losses.mean()

    model.train()
    return out

# Model Class 

print('model def')

class BigramLM(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # backprop based lookup table

  def forward(self, idx, targets=None):

    logits = self.token_embedding_table(idx) # (B,T,C), channels is just vocab_size here

    if targets is None:
      loss = None
    else:

      # shape stuff
      #print(logits.shape)
      #print(targets.shape)

      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)

      # shape stuff
      #print(logits.shape)
      #print(targets.shape)

      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):

    for _ in range(max_new_tokens): # idx is B, T, so like one tensor input, gets genned further, I assume VLLM and quiet-star split this somehow
      
      logits, loss = self(idx) # new preds
      logits = logits[:, -1, :] # switchs it to (B, C), basically the same as the previous view, but only 1 tensor anyway so not B*T
      probs = F.softmax(logits, dim=-1) # (B, C) get the distribution

      # print(probs) # shape/visib stuff

      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1), picks 1 from the distribution
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1), adds it on to the token chain

    return idx

# Model init

print("model init")
model = BigramLM(vocab_size)
m  = model.to(device)

# training loop

xb,yb = get_batch('train')

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) # attached model here

for iter in range(max_iters):

    if iter % eval_iters == 0:

        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")


    xb,yb = get_batch('train')
    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)

    loss.backward()
    optimizer.step()
    
print('model trained')

# making a generation (post training)
idx = torch.zeros((1,1), dtype=torch.long) # a zero (space)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))
