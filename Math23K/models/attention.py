import torch
from torch import nn
# from torch.nn.modules.transformer import TransformerDecoderLayer
from transformers.activations import gelu_new as gelu_bert


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, tensor):
        return tensor.squeeze(dim=self.dim)

    def extra_repr(self):
        return 'dim={dim}'.format(**self.__dict__)

class AveragePooling(nn.Module):
    def __init__(self, dim=-1, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, tensor):
        return tensor.mean(dim=self.dim, keepdim=self.keepdim)

    def extra_repr(self):
        return 'dim={dim}, keepdim={keepdim}'.format(**self.__dict__)

class MultiheadAttentionWeights(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.linear_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dim_head = self.hidden_dim // self.num_heads
        self.sqrt_dim = self.dim_head ** 0.5

    def forward(self, query, key=None, key_ignorance_mask=None, attention_mask=None, head_at_last=True):
        if key is None:
            key = query

        query_len = query.shape[1]
        key_len = key.shape[1]
        batch_size = max(key.shape[0], query.shape[0])

        query = self.linear_q(query)
        key = self.linear_k(key)
        query = query / self.sqrt_dim

        if query.shape[0] == 1:
            query = query.expand(batch_size, -1, -1)
        if key.shape[0] == 1:
            key = key.expand(batch_size, -1, -1)

        query = query.view(batch_size, query_len, self.num_heads, self.dim_head).transpose(1, 2).flatten(0, 1).contiguous()
        key = key.view(batch_size, key_len, self.num_heads, self.dim_head).permute(0, 2, 3, 1).flatten(0, 1).contiguous()
        attention_weights = torch.bmm(query, key).view(batch_size, self.num_heads, query_len, key_len).contiguous()

        if attention_mask is not None:
            attention_weights.masked_fill_(attention_mask.bool(), -1e12)

        if key_ignorance_mask is not None:
            attention_weights.masked_fill_(key_ignorance_mask.unsqueeze(1).unsqueeze(1).bool(), -1e12)

        if head_at_last:
            return attention_weights.permute(0, 2, 3, 1).contiguous()
        else:
            return attention_weights

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiheadAttentionWeights(config)
        self.dropout_p = config.attention_probs_dropout_prob
        self.dropout_attn = nn.Dropout(self.dropout_p)
        self.linear_v = nn.Linear(self.attn.hidden_dim, self.attn.hidden_dim)
        self.linear_out = nn.Linear(self.attn.hidden_dim, self.attn.hidden_dim)

    def forward(self, query, key_value=None, key_ignorance_mask=None, attention_mask=None, return_weights=False):
        if key_value is None:
            key_value = query

        attn_weights = self.attn(query=query, key=key_value, key_ignorance_mask=key_ignorance_mask, attention_mask=attention_mask, head_at_last=False)
        batch_size, _, query_len, key_len = attn_weights.shape
        attn = attn_weights.softmax(dim=-1)
        attn = self.dropout_attn(attn)
        attn = attn.masked_fill(torch.isnan(attn), 0.0).view(-1, query_len, key_len)
        value_size = key_value.shape[0]
        value = self.linear_v(key_value).view(value_size, key_len, self.attn.num_heads, self.attn.dim_head).transpose(1, 2)

        if value_size == 1:
            value = value.expand(batch_size, -1, -1, -1)

        value = value.flatten(0, 1).contiguous()
        output = torch.bmm(attn, value) \
            .view(batch_size, self.attn.num_heads, query_len, self.attn.dim_head) \
            .transpose(1, 2).flatten(2, 3).contiguous()

        output = self.linear_out(output)

        if return_weights:
            return output, attn_weights.permute(0, 2, 3, 1).contiguous()
        else:
            return output


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiheadAttention(config)
        self.mem = MultiheadAttention(config)
        self.dropout_attn = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_mem = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_expand = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_out = nn.Dropout(config.hidden_dropout_prob)
        self.lin_expand = nn.Linear(config.hidden_size, config.intermediate_size)
        self.lin_collapse = nn.Linear(config.intermediate_size, config.hidden_size)
        self.norm_attn = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm_mem = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm_out = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, target, target_ignorance_mask=None, target_attention_mask=None, memory=None, memory_ignorance_mask=None):
        attented = self.attn(query=target, attention_mask=target_attention_mask, key_ignorance_mask=target_ignorance_mask)
        target = target + self.dropout_attn(attented)
        target = self.norm_attn(target)

        if memory is not None:
            attented = self.mem(query=target, key_value=memory, key_ignorance_mask=memory_ignorance_mask)
            target = target + self.dropout_mem(attented)
            target = self.norm_mem(target)

        output = self.lin_collapse(self.dropout_expand(gelu_bert(self.lin_expand(target))))
        target = target + self.dropout_out(output)
        target = self.norm_out(target)

        return target
