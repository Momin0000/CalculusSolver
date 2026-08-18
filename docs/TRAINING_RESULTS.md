# Training Results

**Git Commit Hash:** `c16c7c9963834c864b98c44d18f77acae8cf757d`
**Best Validation Loss:** 0.0149
**Total Epochs Run:** 5

## Per-Epoch Metrics

| Epoch | Train Loss | Val Loss | Per-Token Acc | Val Seq Acc | Saved |
|-------|-----------|----------|---------------|-------------|-------|
| 1 | 1.2289 | 0.2352 | 0.9384 | 0.2080 | Yes |
| 2 | 0.0951 | 0.0417 | 0.9849 | 0.8426 | Yes |
| 3 | 0.0312 | 0.0212 | 0.9928 | 0.9174 | Yes |
| 4 | 0.0207 | 0.0169 | 0.9940 | 0.9272 | Yes |
| 5 | 0.0178 | 0.0149 | 0.9948 | 0.9391 | Yes |

## Configuration Snapshot

- **Learning Rate:** 0.0001
- **Warmup Steps:** 1000
- **Batch Size:** 32
- **Hidden Dim:** 256
- **Max Len:** 48
- **Max Steps/Epoch:** 500
- **Early Stopping:** patience=15, min_delta=0.0002
- **Vocab Size:** 124
- **Gradient Clipping:** max_norm=1.0
- **Rule Prediction:** Multi-head prediction output (decoder_logits, rule_logits, verifier_logits)
