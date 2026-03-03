def generate_block_mask(batch_size, seq_len, block_size):
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    for b in range(batch_size):
        start = random.randint(0, seq_len - block_size)
        mask[b, start:start+block_size] = True
        
    return mask