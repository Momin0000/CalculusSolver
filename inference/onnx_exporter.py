"""
ONNX Model Exporter for CalculusSolver PyTorch weights.
Converts best.pt / model.pkl into best.onnx for lightweight CPU inference.
"""

import os
import sys
import torch
from pathlib import Path


def export_to_onnx(
    checkpoint_path: str = "checkpoints/final/best.pt",
    output_path: str = "checkpoints/final/best.onnx",
    vocab_path: str = "tokenizer/vocab.json"
) -> str:
    """Export PyTorch CalculusModel state dict to ONNX format."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"PyTorch checkpoint not found: {checkpoint_path}")

    from model.architecture import CalculusModel
    from inference.beam_search import load_vocab

    vocab_map = load_vocab(vocab_path)
    vocab_size = len(vocab_map["token_to_id"])
    rule_labels = [k.replace("RULE:", "") for k in vocab_map.get("rule_tokens", {}).keys()]

    model = CalculusModel(
        vocab_size=vocab_size,
        rule_labels=rule_labels,
        hidden_dim=512,
        num_heads=8,
        num_layers=8,
        ffn_dim=2048,
        dropout=0.0
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy inputs for tracing
    batch_size = 1
    seq_len = 256
    dummy_src = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    dummy_positions = torch.zeros((batch_size, seq_len, 3), dtype=torch.float32)
    dummy_pairs = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_src, dummy_positions, dummy_pairs),
        output_path,
        input_names=["src_tokens", "positions", "parent_child_pairs"],
        output_names=["logits", "rule_logits"],
        dynamic_axes={
            "src_tokens": {0: "batch_size", 1: "seq_len"},
            "positions": {0: "batch_size", 1: "seq_len"},
            "parent_child_pairs": {0: "batch_size", 1: "seq_len", 2: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"}
        },
        opset_version=14
    )

    print(f"[ONNX Export] Model successfully exported to: {output_path}")
    return output_path


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/final/best.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/final/best.onnx"
    try:
        export_to_onnx(ckpt, out)
    except Exception as exc:
        print(f"Export skipped/failed: {exc}")
