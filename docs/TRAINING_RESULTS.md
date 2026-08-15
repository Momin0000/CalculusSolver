# Training Results

**Git Commit Hash:** `d26bbe19f4b0b694d6bca560b51cd60e42a36cbc`
**Best Validation Loss:** 0.0167
**Total Epochs Run:** 10

## Per-Epoch Metrics

| Epoch | Train Loss | Val Loss | Per-Token Acc | Val Seq Acc | Saved |
|-------|-----------|----------|---------------|-------------|-------|
| 1 | 0.2888 | 0.0200 | 0.9856 | 0.7683 | Yes |
| 2 | 0.0191 | 0.0170 | 0.9898 | 0.8346 | Yes |
| 3 | 0.0179 | 0.0170 | 0.9898 | 0.8345 | No |
| 4 | 0.0176 | 0.0202 | 0.9892 | 0.8252 | No |
| 5 | 0.0173 | 0.0167 | 0.9899 | 0.8363 | Yes |
| 6 | 0.0172 | 0.0171 | 0.9898 | 0.8346 | No |
| 7 | 0.0170 | 0.0168 | 0.9899 | 0.8363 | No |
| 8 | 0.0170 | 0.0173 | 0.9897 | 0.8335 | No |
| 9 | 0.0171 | 0.0167 | 0.9898 | 0.8356 | No |
| 10 | 0.0168 | 0.0169 | 0.9898 | 0.8356 | No |

## Configuration Snapshot

- **Architecture:** SimpleCalculusModel (standard nn.Transformer encoder-decoder)
- **Learning Rate:** 0.0001
- **Warmup Steps:** 1000
- **Batch Size:** 32
- **Hidden Dim:** 256
- **Max Steps/Epoch:** 3500
- **Early Stopping:** patience=12, min_delta=0.0002
- **Vocab Size:** 124
- **Gradient Clipping:** max_norm=1.0
- **Rule Prediction:** Folded into output sequence as leading RULE:xxx token
