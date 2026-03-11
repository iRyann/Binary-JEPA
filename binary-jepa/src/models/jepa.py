import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoder import Conv1DEncoder
from src.models.predictor import Predictor


class IJEPA(nn.Module):
    MASK_TOKEN_ID = 2

    def __init__(self, vocab_size, dim=256):
        super().__init__()

        self.context_encoder = Conv1DEncoder(vocab_size, out_dim=dim)
        self.target_encoder = Conv1DEncoder(vocab_size, out_dim=dim)
        self.predictor = Predictor(dim)

        # initialisation -> copie de l'encodeur de contexte
        self._update_target_encoder(0)

    @torch.no_grad()
    def _update_target_encoder(self, m=0.996):
        for param_q, param_k in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data = m * param_k.data + (1 - m) * param_q.data

    def forward(
        self, x: torch.Tensor, input_mask: torch.Tensor, pred_mask: torch.Tensor
    ):
        """
        x: (batch_size, sequence_length) batch d'entrée
        input_mask: (batch_size, sequence_length) masque pour occulter les blocs 2 et 3 (pré-cible et cible)
        """

        x_masked = x.clone()
        x_masked[input_mask] = self.MASK_TOKEN_ID

        z_context = self.context_encoder(x_masked)

        with torch.no_grad():
            z_target = self.target_encoder(x)

        # prédiction des zones masquées uniquement
        z_pred = self.predictor(z_context)

        loss = F.smooth_l1_loss(z_pred[pred_mask], z_target[pred_mask])

        return loss
