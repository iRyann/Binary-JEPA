import torch
from src.models.jepa import IJEPA

def generate(
        x, 
        pad_token_id: int = IJEPA.PAD_TOKEN_ID
             ) -> tuple[torch.Tensor,torch.Tensor]:
    real_lengths = (x != pad_token_id).sum(dim=1)

    input_mask = torch.zeros_like(x, dtype=torch.bool)
    pred_mask = torch.empty_like(input_mask).copy_(input_mask)

    for i in range(x.shape[0]):
        # tiers 2 et 3 masqués en entrée
        start = (real_lengths[i]*2)//3
        end = real_lengths[i]
        input_mask[i,start:end] = True

        pred_mask[i,end-1] = True
            
    return input_mask,pred_mask