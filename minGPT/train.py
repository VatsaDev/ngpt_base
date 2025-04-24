import torch
import torch.nn as nn
from torch.nn import functional as F 

# hyperparams 
batch_size = 64
block_size = 128 # ctx_len
eval_interval = 20
lr = 3e-4

max_iters = 200
eval_iters = 20

n_embd = 32
n_head = 4
n_layer = 4
dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

# data file
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# tokenizer

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# tokenized train, val 
data = torch.tensor(encode(text))
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Model

class Head(nn.Module):

    "one self attn head"

    def __init__(self, head_size):

        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape

        # qkv

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # actual attn 

        wei = q @ k.transpose(-2, -1) * C ** -0.5 # scaled dot prod
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # value weights

        out = wei @ v 
        return out

class MultiHeadAttn(nn.Module):

    def __init__(self, num_heads, head_size):

        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # projection adds computation back to the residual pathway
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 

        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)

        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd), # also a residual projection
            nn.Dropout(dropout) # dropout improves stability
        )

    def forward(self, x):

        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):

        super().__init__()

        head_size = n_embd // n_head
        self.sa = MultiHeadAttn(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # backprop-based lookup table
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # positional encoding
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        if targets is None:

            loss = None

        else:

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)

            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


# Model init
model = Transformer()
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M params')


# training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range((max_iters+1)):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print('model trained')

# making a generation (post training)
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(idx, max_new_tokens=900)[0].tolist()))

