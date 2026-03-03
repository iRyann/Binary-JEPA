import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoder import Conv1DEncoder
from src.models.predictor import Conv1DEncoder

class IJEPA(nn.Module):
    def __init__(self, vocab_size, dim=256):
        super().__init__()

        MASK_TOKEN_ID = 2
        
        self.context_encoder = Conv1DEncoder(vocab_size, out_dim=dim)
        self.target_encoder = Conv1DEncoder(vocab_size, out_dim=dim)
        self.predictor = Predictor(dim)

        # initialisation -> copie de l'encodeur de contexte
        self._update_target_encoder(0)

    @torch.no_grad()
    def _update_target_encoder(self, m=0.996):
        for param_q, param_k in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_k.data = m * param_k.data + (1 - m) * param_q.data

    def forward(self, x, mask):
        """
        x: (B, L)
        mask: (B, L) masque pour occulter les blocs 2 et 3 (pré-cible et cible)
        """
        x_masked = x.copy()
        x_masked[mask] = IJEPA.MASK_TOKEN_ID

        z_context = self.context_encoder(x_masked)
        
        with torch.no_grad():
            z_target = self.target_encoder(x)

        # prédiction des zones masquées uniquement
        z_pred = self.predictor(z_context)

        loss = F.mse_loss(
            z_pred[mask],
            z_target[mask]
        )

        return loss