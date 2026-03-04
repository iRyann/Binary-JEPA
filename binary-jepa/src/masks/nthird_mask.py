import torch
from src.models.jepa import IJEPA

def generate_last_two_thirds_mask(x, batch_size : int, start_third : int, end_third : int, pad_token_id: int = IJEPA.MASK_TOKEN_ID) -> List[int]:
    real_lengths = (x != pad_token_id).sum(dim=1)

    mask = torch.zeros_like(x, dtype=torch.bool)

    for i in range(batch_size):
        # Début du masquage = 1/3 de la séquence
        start = real_lengths[i]*start_third // 3
        end = real_length[i]*end_third // 3
        
        # Masquer uniquement la partie réelle
        mask[i,start:end] = True
    
    return mask