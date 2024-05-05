import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCausalAttention(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 max_T: int,
                 n_heads: int,
                 dropout: float
    ):
        super(MaskedCausalAttention, self).__init__()
        self.n_heads = n_heads
        self.max_T = max_T

        self.queries = nn.Linear(hidden_dim, hidden_dim)
        self.keys = nn.Linear(hidden_dim, hidden_dim)
        self.values = nn.Linear(hidden_dim, hidden_dim)

        self.projections = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection_dropout = nn.Dropout(dropout)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        N, D = self.n_heads, C // self.n_heads

        q = self.queries(x).view(B, T, N, D).transpose(1, 2)
        k = self.keys(x).view(B, T, N, D).transpose(1, 2)
        v = self.values(x).view(B, T, N, D).transpose(1, 2)

        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        weights = weights.masked_fill(self.mask[..., :T, :T] == 0, float('-inf'))
        normalized_weights = F.softmax(weights, dim=-1)

        attention = self.attention_dropout(normalized_weights @ v)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)

        out = self.projection_dropout(self.projections(attention))
        return out


class TransformerBlock(nn.Module):
    def __init__(self, 
                 hidden_dim: int,
                 max_T: int,
                 n_heads: int,
                 dropout: float
    ):
        super(TransformerBlock, self).__init__()
        self.attention = MaskedCausalAttention(hidden_dim, max_T, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        x = self.ln1(x)
        x = x + self.feed_forward(x)
        x = self.ln2(x)
        return x
    

class DecisionTransformer(nn.Module):
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 n_blocks: int,
                 hidden_dim: int,
                 context_len: int,
                 n_heads: int,
                 continuous: bool = True,
                 dropout: float = 0.2,
                 max_timesteps: int = 4096
    ):
        super(DecisionTransformer, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        input_seq_len = 3 * context_len
        blocks = [TransformerBlock(hidden_dim, input_seq_len, n_heads, dropout) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        self.embed_ln = nn.LayerNorm(hidden_dim)
        self.embed_timestamp = nn.Embedding(max_timesteps, hidden_dim)
        self.embed_rtg = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(state_dim, hidden_dim)

        if continuous:
            self.embed_action = nn.Linear(action_dim, hidden_dim)
            use_action_tanh = True
        else:
            self.embed_action = nn.Embedding(action_dim, hidden_dim)
            use_action_tanh = False
        
        self.predict_rtg = nn.Linear(hidden_dim, 1)
        self.predict_state = nn.Linear(hidden_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_dim, self.action_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )

    
    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, timesteps, states, actions, returns_to_go):
        B, T, _ = states.shape

        time_embeddings = self.embed_timestamp(timesteps)

        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        rtg_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        h = torch.stack(
            (rtg_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.hidden_dim)

        h = self.embed_ln(h)

        h = self.transformer(h)

        h = h.reshape(B, T, 3, self.hidden_dim).permute(0, 2, 1, 3)

        rtg_preds = self.predict_rtg(h[:, 2])
        state_preds = self.predict_state(h[:, 2])
        action_preds = self.predict_action(h[:, 1])

        return state_preds, action_preds, rtg_preds