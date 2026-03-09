import torch
from src.models.jepa import IJEPA

def generate(
        x, 
        batch_size : int,
        pad_token_id: int = IJEPA.MASK_TOKEN_ID
             ) -> tuple[torch.Tensor,torch.Tensor]:
    real_lengths = (x != pad_token_id).sum(dim=1)

    input_mask = torch.zeros_like(x, dtype=torch.bool)
    pred_mask = torch.empty_like(input_mask).copy_(input_mask)

    for i in range(batch_size):
        # tiers 2 et 3 masqués en entrée
        start = (real_lengths[i] * 1)//3
        end = real_lengths[i]
        input_mask[i,start:end] = True

        start = (real_lengths[i] * 2)//3
        pred_mask[i,start:end] = True
        
        # Masquer uniquement la partie réelle
    
    return input_mask,pred_mask