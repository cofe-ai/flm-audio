import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig


class DepthGPTConfig(PretrainedConfig):
    def __init__(
        self,
        block_size: int = 8,
        vocab_size: int = 2049, # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        n_layer: int = 6,
        n_head: int = 16,
        n_embd: int = 1024,
        dropout: float = 0.0,
        bias: bool = False, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        main_hidden_size = 1536,
        pad_token_id = 2048,
        use_cmlp = True,
        use_rmsnorm = False,
        use_swiglu = False
    ):
        """
            {
                "block_size": 8,
                "vocab_size": 2049,
                "n_layer": 6,
                "n_head": 16,
                "n_embd": 1024,
                "dropout": 0.0,
                "bias": false,
                "main_hidden_size": 1536,
                "pad_token_id": 2048,
                "use_cmlp": true
            }
        """
        # super().__init__(**kwargs)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.main_hidden_size = main_hidden_size
        self.pad_token_id = pad_token_id
        self.use_cmlp = use_cmlp
        self.use_rmsnorm = use_rmsnorm
        self.use_swiglu = use_swiglu

################################################################################################
#                                   GPT style
################################################################################################

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

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
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class MLP_swiglu(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = int(8 * config.n_embd / 3)
        self.gate_proj = nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(self.intermediate_size, config.n_embd, bias=config.bias)
        self.act_fn = F.silu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd) if config.use_rmsnorm else LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd) if config.use_rmsnorm else LayerNorm(config.n_embd, bias=config.bias)
        mlp_cls = MLP_swiglu if config.use_swiglu else MLP
        self.mlp = mlp_cls(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BlockCMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.channel_size = config.block_size
        self.ln_1 = RMSNorm(config.n_embd) if config.use_rmsnorm else LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd) if config.use_rmsnorm else LayerNorm(config.n_embd, bias=config.bias)
        mlp_cls = MLP_swiglu if config.use_swiglu else MLP
        self.mlps = nn.ModuleList([mlp_cls(config) for _ in range(self.channel_size)])

        assert self.channel_size == 8, f"DEBUG, self.channel_size={self.channel_size} != 8"

    def forward(self, x):
        _, channel_size, _ = x.shape
        # assert channel_size == self.channel_size
        x = x + self.attn(self.ln_1(x))

        xl = self.ln_2(x)
        x = x + torch.cat(
            [self.mlps[c](xl[:, c:c+1, :]) for c in range(self.channel_size)],
            dim=1
        )
        return x


class DepthGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.num_channel = config.block_size

        self.linear_in = nn.Linear(config.main_hidden_size, config.n_embd * config.block_size, bias=False)

        block_cls = BlockCMLP if config.use_cmlp else Block
        self.transformer = nn.ModuleDict(dict(
            wtes = nn.ModuleList([nn.Embedding(config.vocab_size, config.n_embd) for _ in range(self.num_channel)]),
            wpe = nn.Embedding(self.num_channel, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([block_cls(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd) if config.use_rmsnorm else LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_heads = nn.ModuleList([nn.Linear(config.n_embd, config.vocab_size, bias=False) for _ in range(self.num_channel)])

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=False):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,
                main_hidden_states, # [seq, main_dim]
                audio_token_ids # [seq, 7]
            ):

        assert main_hidden_states.shape[0] == audio_token_ids.shape[0]
        in_audio_token_num = audio_token_ids.shape[-1]

        device = audio_token_ids.device

        audio_token_ids = F.pad(audio_token_ids, (1, 0), value=self.config.pad_token_id)

        x = torch.stack(
            [self.transformer.wtes[c](audio_token_ids[:, c]) for c in range(in_audio_token_num + 1)]
        ).transpose(0, 1)  # [seq, in_audio_token_num]

        x += self.transformer.wpe(
            torch.arange(0, in_audio_token_num + 1, dtype=torch.long, device=device)
        ).unsqueeze(0) # position embeddings of shape (1, 8, depth_dim)

        main_hidden = self.linear_in(main_hidden_states).view(main_hidden_states.shape[0], self.config.block_size, -1)[:, :in_audio_token_num+1, :]
        x += main_hidden

        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)

        # [seq, 8, hidden]
        x = self.transformer.ln_f(x)

        # [seq, 8, hidden] (linear)-> [8, seq, vocab]
        x = torch.stack([self.lm_heads[c](x[:, c, :]) for c in range(x.shape[1])])

        # [8, seq, vocab] -> [seq, 8, vocab]
        x = x.transpose(0,1)

        return x

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


if __name__ == "__main__":
    config = {
    "bias": False,
    "dropout": 0.0,
    "n_embd": 1024,
    "n_head": 16,
    "n_layer": 6,
    "use_cmlp": True,
    "use_rmsnorm": True,
    "use_swiglu": True,
    "main_hidden_size": 4096
    }
    model_config = DepthGPTConfig(**config)
    model = DepthGPT(config=model_config)

    main_hidden_states = torch.rand((1, 4096))
    decoded_audio_tokens = torch.empty((1, 0), dtype=torch.long, device=main_hidden_states.device)
    outputs = model(main_hidden_states, decoded_audio_tokens)
