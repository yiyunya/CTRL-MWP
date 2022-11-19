import torch
from torch import nn


def gather_vectors(encoded, num_ids):
    batch_size, seq_len, hidden_size = encoded.shape
    _, max_len = num_ids.shape

    gathered = torch.zeros(batch_size, max_len, hidden_size, dtype=encoded.dtype, device=encoded.device)
    for row in range(batch_size):
        for i in range(max_len):
            if num_ids[row][i] > -1:
                gathered[row, i] = encoded[row, num_ids[row][i]]
            else:
                break

    return gathered

class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, text_ids, text_pads, num_ids, num_pads):
        outputs = self.model(input_ids=text_ids, attention_mask=text_pads)
        encoded = outputs[0]
        num = gather_vectors(encoded, num_ids)
        return {'text': encoded, 'text_pads': text_pads, 'num': num, 'num_pads': num_pads}

    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)