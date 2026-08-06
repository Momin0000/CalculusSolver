# Training Results

**Best Validation Loss:** 0.0355
**Total Epochs Run:** 9

## Per-Epoch Metrics

| Epoch | Train Loss | Val Loss | Val Seq Accuracy | Checkpoint Saved |
|-------|-----------|----------|-------------------|-----------------|
| 1 | 0.0955 | 0.0355 | 0.7459 | Yes |
| 2 | 0.0362 | 0.0355 | 0.7459 | No |
| 3 | 0.0354 | 0.0353 | 0.7459 | No |
| 4 | 0.0352 | 0.0352 | 0.7459 | No |
| 5 | 0.0353 | 0.0353 | 0.7459 | No |
| 6 | 0.0351 | 0.0352 | 0.7459 | No |
| 7 | 0.0350 | 0.0353 | 0.7459 | No |
| 8 | 0.0349 | 0.0352 | 0.7459 | No |
| 9 | 0.0347 | 0.0354 | 0.7459 | No |

## Configuration

- **Architecture:** SimpleCalculusModel (standard nn.Transformer encoder-decoder)
- **Learning Rate:** 0.0001
- **Batch Size:** 32
- **Hidden Dim:** 256
- **Max Steps/Epoch:** 3500
- **Early Stopping:** patience=8, min_delta=0.0005
- **Vocab Size:** 105
- **Gradient Clipping:** max_norm=1.0
- **Rule prediction:** folded into output sequence as leading RULE:xxx token (see docs/KNOWN_ISSUES.md)
