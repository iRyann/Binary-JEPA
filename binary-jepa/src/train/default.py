import torch
from src.models.jepa import IJEPA
import src.masks.noiseproof as mask
import torch.utils.data.Dataloader as DataLoader
from tqdm import tqdm

def main():
    data = []

    model = IJEPA(vocab_size=5000, dim=256) # obtenir la taille du vocab
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    data = DataLoader(dataset, batch_size=10, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)

    for batch in tqdm(data): # obtenir le jeu de données, sous quelle forme ?

        input_mask, pred_mask = mask.generate(batch.shape[0])

        loss = model(batch,input_mask,pred_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model._update_target_encoder()

if __name__ == "__main__":
    main()