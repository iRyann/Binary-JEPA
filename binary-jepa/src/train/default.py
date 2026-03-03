from src.masks.default import generate_block_mask
from src.models.jepa import IJEPA

def main():
    # model = IJEPA(vocab_size=5000, dim=256)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # for batch in dataloader:
    #     input_ids = batch  # (B, L)

    #     mask = generate_block_mask(
    #         batch_size=input_ids.size(0),
    #         seq_len=input_ids.size(1),
    #         block_size=32
    #     )

    #     loss = model(input_ids, mask)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     model._update_target_encoder()

if __name__ == "__main__":
    main()