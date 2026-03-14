import torch
from src.models.jepa import IJEPA
from random import randint

def generate(
        x, 
        pad_token_id: int = IJEPA.PAD_TOKEN_ID
             ) -> tuple[torch.Tensor,torch.Tensor]:
    real_lengths = (x != pad_token_id).sum(dim=1)

    input_mask = torch.zeros_like(x, dtype=torch.bool)

    for i in range(x.shape[0]):
        masked_index = randint(0,real_lengths[i]-1)
        input_mask[i,masked_index] = True
            
    return input_mask,input_mask