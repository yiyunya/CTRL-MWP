import math
import torch
from torch import nn
from torch.nn import functional as F
from .attention import MultiheadAttentionWeights, MultiheadAttention


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        div_term = (torch.arange(0, embedding_dim) // 2) * 2
        div_term = torch.exp(div_term.float() * (-math.log(10000.0) / embedding_dim))
        multiplier = torch.zeros(2, embedding_dim, dtype=torch.float)
        multiplier[0, 1::2] = 1.0 
        multiplier[1, 0::2] = 1.0

        self.register_buffer('_div_term', div_term)
        self.register_buffer('multiplier', multiplier)

    @property
    def device(self):
        return self._div_term.device

    def before_trigonometric(self, indices):
        indices = indices.float()
        return indices * self._div_term

    def forward(self, index_or_range, ignored_index=-1):
        with torch.no_grad():
            return self._forward(index_or_range, ignored_index)

    def _forward(self, index_or_range, ignored_index=-1):
        if type(index_or_range) is int:
            indices = torch.arange(0, index_or_range)
        elif type(index_or_range) is range:
            indices = torch.as_tensor(list(index_or_range))
        else:
            indices = index_or_range

        indices = indices.unsqueeze(-1)
        indices = indices.to(self.device)
        phase = self.before_trigonometric(indices)
        cos_value = phase.cos()
        sin_value = phase.sin()
        cos_multiplier = self.multiplier[0]
        sin_multiplier = self.multiplier[1]
        result_shape = [1] * (phase.dim() - 1) + [-1]
        cos_multiplier = cos_multiplier.view(*result_shape)
        sin_multiplier = sin_multiplier.view(*result_shape)
        result = cos_value * cos_multiplier + sin_value * sin_multiplier
        ignored_indices = (indices == ignored_index)
        if ignored_indices.any():
            result.masked_fill_(ignored_indices, 0.0)
        return result.contiguous()


def get_embedding_without_pad(embedding, tokens, ignore_index=-1):
    tokens = tokens.clone()
    ignore_positions = (tokens == ignore_index)
    if ignore_positions.any():
        tokens.masked_fill_(ignore_positions, 0)

    if isinstance(embedding, nn.Embedding):
        embedding = embedding(tokens)
    else:
        embedding = F.embedding(tokens, embedding)

    if ignore_positions.any():
        embedding.masked_fill_(ignore_positions.unsqueeze(-1), 0.0)

    return embedding.contiguous()


def apply_module_dict(modules, encoded, **kwargs):
    output = encoded
    keys = sorted(modules.keys())
    for key in keys:
        layer = modules[key]
        if isinstance(layer, (MultiheadAttentionWeights, MultiheadAttention)):
            output = layer(query=output, **kwargs)
        else:
            output = layer(output)

    return output


def apply_across_dim(function, dim=1, shared_keys=None, **tensors):
    shared_arguments = {}
    repeat_targets = {}
    for key, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or (shared_keys and key in shared_keys):
            shared_arguments[key] = tensor
        else:
            repeat_targets[key] = tensor

    size = {key: tensor.shape[dim] for key, tensor in repeat_targets.items()}
    assert len(set(size.values())) == 1, 'Tensors does not have same size on dimension %s: We found %s' % (dim, size)
    size = list(size.values())[0]
    output = {}

    for i in range(size):
        kwargs = {key: tensor.select(dim=dim, index=i).contiguous() for key, tensor in repeat_targets.items()}
        kwargs.update(shared_arguments)
        for key, tensor in function(**kwargs).items():
            if key in shared_keys:
                continue

            if key not in output:
                output[key] = []

            output[key].append(tensor.unsqueeze(dim=dim))

    assert all(len(t) == size for t in output.values())
    return {key: torch.cat(tensor, dim=dim).contiguous() for key, tensor in output.items()}