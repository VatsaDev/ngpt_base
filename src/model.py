import torch
import torch.nn as nn
from torch.nn import functional as F 

import inspect

# add Rope, local attn, etc

config = {
    "n_embd": 32,
    "n_head": 4,
    "n_layer": 4,
    "dropout": 0.2,
    "vocab_size": 65,
    "ctx_len": 128,
    "device": "cpu"
}

class CasualSelfAttn(nn.Module):

    def __init__(self):

        super().__init__()

        assert config['n_embd'] % config['n_head'] == 0 # needed, the attn has to be split evenly and completely across all heads 

        self.c_attn = nn.Linear(config['n_embd'], 3*config['n_embd'], bias=False) # qkv combined
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'], bias=False) # out projection 

        # regularization stuff

        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])

        self.n_embd = config['n_embd']
        self.n_head = config['n_head']
        self.dropout = config['dropout']

        # flash attn, gpus go brrrrr

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:

            print("imagine not having self attn") # NOOOOOO cant have second half in a real codebase

            self.register_buffer("bias", torch.tril(torch.ones(config['ctx_len'], config['ctx_len']))
                                        .view(1, 1, config['ctx_len'], config['ctx_len'])) # attn mask

    def forward(self, x):

        B, T, C = x.size()

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attn

            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, n_head, T, T) x (B, n_head, T, hs) -> (B, n_head, T, head_size)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs
        
        y = self.resid_dropout(self.c_proj(y)) # ouput proj

        return y

class MLP(nn.Module):

    def __init__(self, n_embd):

        super().__init__()
        
        self.c_fc    = nn.Linear(config['n_embd'], 4 * config['n_embd'])
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config['n_embd'], config['n_embd'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):

        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x

class Block(nn.Module):

    def __init__(self, n_embd, n_head):

        super().__init__()

        head_size = config['n_embd'] // config['n_head']
        self.attn = CasualSelfAttn()
        self.mlp = MLP(config['n_embd'])
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])

    def forward(self, x):

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x

class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config['vocab_size'], config['n_embd'])  # backprop-based lookup table
        self.position_embedding_table = nn.Embedding(config['ctx_len'], config['n_embd'])  # positional encoding
        self.blocks = nn.Sequential(*[Block(config['n_embd'], n_head=config['n_head']) for _ in range(config['n_layer'])])
        self.ln_f = nn.LayerNorm(config['n_embd']) # final layer norm 
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'])

        self.token_embedding_table.weight = self.lm_head.weight # weight tying, small perf boost and saved like 2k params

        self.apply(self._init_weights) # normal init is nice

    
    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=config['device']))  # (T, C)
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

            assert max_new_tokens < config['ctx_len'] # just an extra check
            
            idx_cond = idx[:, -config['ctx_len']:]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)

            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx

    # helper methods, not key parts of arch

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def configure_optimizers(self, weight_decay, learning_rate, betas, device):

        param_dict = {pn: p for pn, p in self.named_parameters()} # start with all params
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} # filter non-grad

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
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    def estimate_mfu(self, params, fwdbwd_per_iter, dt):

        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = params
        L, H, Q, T = config['n_layer'], config['n_head'], config['n_embd']//config['n_head'], config['ctx_len']
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of T4 fp16 peak
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 65e12 # 65 tflops
        mfu = flops_achieved / flops_promised
        return mfu
