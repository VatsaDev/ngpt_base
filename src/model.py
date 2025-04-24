import torch
import torch.nn as nn
from torch.nn import functional as F
import math # Added for manual attention calculation
import inspect

# This global config dictionary will be OVERWRITTEN by the training script
# before the Transformer class is instantiated.
config = {
    "n_embd": 1024,      # Default, will be updated
    "n_head": 8,       # Default, will be updated
    "n_layer": 8,      # Default, will be updated
    "dropout": 0.0,     # Default, will be updated (though 0.2 is in your example)
    "vocab_size": 50257,# Default, will be updated
    "ctx_len": 1024,    # Default, will be updated (alias for block_size)
    "bias": True,       # Added from previous script's example
    # "device": "cpu"   # REMOVED - device determined dynamically
}

class CasualSelfAttn(nn.Module):

    def __init__(self):
        super().__init__()
        # Assumes config has been updated by the training script before instantiation
        assert config['n_embd'] % config['n_head'] == 0

        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'], bias=config.get('bias', False)) # Use bias from config
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'], bias=config.get('bias', False)) # Use bias from config

        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])

        self.n_embd = config['n_embd']
        self.n_head = config['n_head']
        self.dropout = config['dropout']
        # Renamed ctx_len to block_size for consistency with training script
        self.block_size = config['ctx_len']

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(self.block_size, self.block_size))
                                        .view(1, 1, self.block_size, self.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, n_embd): # n_embd passed to potentially override config if needed later
        super().__init__()
        # Assumes config has been updated by the training script
        self.c_fc    = nn.Linear(config['n_embd'], 4 * config['n_embd'], bias=config.get('bias', False)) # Use bias from config
        # GELU activation - using approximation for potential speedup if needed
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config['n_embd'], config['n_embd'], bias=config.get('bias', False)) # Use bias from config
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    # Removed n_embd, n_head args as they directly use the global config
    def __init__(self):
        super().__init__()
        # Assumes config has been updated by the training script
        # LayerNorm needs the embedding dimension
        self.ln_1 = nn.LayerNorm(config['n_embd'], bias=config.get('bias', False)) # Use bias from config
        self.attn = CasualSelfAttn()
        self.ln_2 = nn.LayerNorm(config['n_embd'], bias=config.get('bias', False)) # Use bias from config
        self.mlp = MLP(config['n_embd']) # Pass n_embd in case MLP needs it internally

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        # Assumes config has been updated by the training script before instantiation
        assert config['vocab_size'] is not None
        assert config['ctx_len'] is not None # Renamed to block_size internally
        self.block_size = config['ctx_len']

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['n_embd']),
            wpe = nn.Embedding(self.block_size, config['n_embd']),
            drop = nn.Dropout(config['dropout']),
            h = nn.ModuleList([Block() for _ in range(config['n_layer'])]),
            ln_f = nn.LayerNorm(config['n_embd'], bias=config.get('bias', False)), # Use bias from config
        ))
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config['n_layer']))

        print(f"Model initialized with {sum(p.numel() for p in self.parameters())/1e6:.2f}M parameters")


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # --- MINIMAL CHANGE 1: Get device from input ---
        device = idx.device
        # --- END CHANGE 1 ---
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        # --- MINIMAL CHANGE 2: Use dynamically determined device ---
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        # --- END CHANGE 2 ---

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward final layer norm and lm_head on the very last position
            # Note: This optimization is not needed if using generate method below
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    # --- MINIMAL CHANGE 3: Update generate method ---
    @torch.no_grad() # Ensure no gradients are computed during generation
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    # --- END CHANGE 3 ---


    # --- MINIMAL CHANGE 4: Update configure_optimizers signature ---
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type): # Changed 'device' to 'device_type'
    # --- END CHANGE 4 ---

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # --- MINIMAL CHANGE 5: Use device_type ---
        use_fused = fused_available and device_type.startswith('cuda') # Check prefix for 'cuda'
        # --- END CHANGE 5 ---
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    # --- MINIMAL CHANGE 6: Update estimate_mfu to use instance config ---
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters()) # Use actual parameter count
        cfg = self.transformer.h[0].attn # Get config from an attention block instance if needed
                                         # Or better, access directly if stored on self or use model.config
                                         # For now, let's assume model.config is globally updated
        L, H, Q, T = config['n_layer'], config['n_head'], config['n_embd']//config['n_head'], self.block_size

        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak FLOPS
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        # Adjust peak flops based on hardware if necessary
        # A100 = 312 TFLOPS (BF16), H100 = 989 TFLOPS (FP16) / 495 TFLOPS (TF32)
        # T4 = 65 TFLOPS (FP16)
        flops_promised = 312e12 # A100 BF16 peak flops
        if torch.cuda.is_available():
             dev_prop = torch.cuda.get_device_properties(torch.cuda.current_device())
             if dev_prop.major >= 8: # Ampere or newer
                 # Use BF16 peak for A100
                 flops_promised = 312e12
                 if dev_prop.major >= 9: # Hopper
                      # Using FP16 peak for H100, adjust if using TF32
                      flops_promised = 989e12
             elif dev_prop.major == 7: # Volta/Turing (like T4)
                 flops_promised = 65e12 # T4 FP16 peak

        mfu = flops_achieved / flops_promised
        return mfu
    # --- END CHANGE 6 ---
