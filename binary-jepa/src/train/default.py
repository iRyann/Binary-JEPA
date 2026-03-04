import torch
from src.models.jepa import IJEPA
import src.masks.noiseproof as mask


def main():
    data = []

    model = IJEPA(vocab_size=5000, dim=256) # obtenir la taille du vocab
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for batch in data: # obtenir le jeu de données, sous quelle forme ?

        input_mask, pred_mask = mask.generate(batch.shape[0])

        loss = model(batch,input_mask,pred_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model._update_target_encoder()

if __name__ == "__main__":
    main()