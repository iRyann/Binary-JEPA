import torch
from random import randint

def generate(
        batch_size, 
        seq_len, 
        block_size
    ) -> tuple[torch.Tensor,torch.Tensor]:
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    for b in range(batch_size):
        start = randint(0, seq_len - block_size)
        mask[b, start:start+block_size] = True
        
    return mask,mask