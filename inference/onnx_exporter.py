"""
ONNX Model Exporter for CalculusSolver PyTorch weights.
Converts best.pt into best.onnx for lightweight CPU inference.

Rewritten to target model/simple_transformer.py's SimpleCalculusModel
(the architecture actually trained since PR #27's pivot), not the old
tree-structured model/architecture.py::CalculusModel. That class expects
src_positions/parent_child_pairs which were never produced by training
and are not part of this model's forward() signature.
"""

import os
import sys
import torch
from pathlib import Path


def export_to_onnx(
    checkpoint_path: str = "checkpoints/final/best.pt",
    output_path: str = "checkpoints/final/best.onnx",
    vocab_path: str = "tokenizer/vocab.json",
    hidden_dim: int = 256,   # must match docs/TRAINING_RESULTS.md's "Hidden Dim" for the checkpoint being exported
    max_len: int = 32,       # must match config.json / TRAINING_RESULTS.md for the checkpoint being exported
) -> str:
    """Export PyTorch SimpleCalculusModel state dict to ONNX format."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"PyTorch checkpoint not found: {checkpoint_path}")

    from model.simple_transformer import SimpleCalculusModel
    from inference.beam_search import load_vocab

    vocab_map = load_vocab(vocab_path)
    vocab_size = len(vocab_map["token_to_id"])
    pad_id = vocab_map["token_to_id"].get("[PAD]", 0)

    model = SimpleCalculusModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        pad_id=pad_id,
        max_len=max_len,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy inputs matching forward(self, src_seq, tgt_in_seq) -- no positions,
    # no parent_child_pairs; this model never took them.
    batch_size = 1
    dummy_src = torch.randint(1, vocab_size, (batch_size, max_len), dtype=torch.long)
    dummy_tgt_in = torch.randint(1, vocab_size, (batch_size, max_len), dtype=torch.long)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_src, dummy_tgt_in),
        output_path,
        input_names=["src_seq", "tgt_in_seq"],
        output_names=["logits"],
        dynamic_axes={
            "src_seq": {0: "batch_size", 1: "seq_len"},
            "tgt_in_seq": {0: "batch_size", 1: "tgt_len"},
            "logits": {0: "batch_size", 1: "tgt_len"},
        },
        opset_version=14,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[ONNX Export] Model successfully exported to: {output_path} ({size_mb:.1f} MB)")
    return output_path


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/final/best.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/final/best.onnx"
    export_to_onnx(ckpt, out)