"""
Exports the trained SimpleCalculusModel (model/simple_transformer.py) to
ONNX for the torch-free production deployment path (Option A -- see
docs/EXPORT_DECISION.md for the measured size comparison vs Vercel's
~250MB serverless limit).

NOTE: model/simple_transformer.py's SimpleCalculusModel is a single
encoder-decoder nn.Transformer with rule prediction folded into the output
sequence (see that file's docstring). There is no separate RuleHead to
export -- unlike the older model/transformer.py design, this model has
exactly one set of weights and one ONNX graph.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.simple_transformer import SimpleCalculusModel
from inference.grammar import load_vocab


def export_to_onnx(
    checkpoint_path: str = os.path.join("checkpoints", "final", "best.pt"),
    output_path: str = os.path.join("deployment", "artifacts", "best.onnx"),
    vocab_path: str = os.path.join("tokenizer", "vocab.json"),
    config_path: str = "config.json",
) -> str:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"PyTorch checkpoint not found: {checkpoint_path}\n"
            "This must be a checkpoint that inference/solve.py can already "
            "load successfully -- if solve.py fails to load it, export will "
            "fail with the identical state_dict mismatch. Confirm with "
            "Developer 3 that this checkpoint is signed off before exporting."
        )
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    vocab_map = load_vocab(vocab_path)
    vocab_size = max(vocab_map["token_to_id"].values()) + 1
    pad_id = vocab_map["token_to_id"]["[PAD]"]

    hidden_dim = 128
    max_len = 32
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
            hidden_dim = cfg.get("hidden_dim", hidden_dim)
            max_len = cfg.get("max_len", max_len)

    model = SimpleCalculusModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        pad_id=pad_id,
        max_len=max_len,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_src = torch.randint(1, vocab_size, (1, max_len), dtype=torch.long)
    dummy_tgt_in = torch.randint(1, vocab_size, (1, max_len), dtype=torch.long)

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
    print(f"[export_onnx] Exported {checkpoint_path} -> {output_path} ({size_mb:.2f} MB)")
    return output_path


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join("checkpoints", "final", "best.pt")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("deployment", "artifacts", "best.onnx")
    export_to_onnx(ckpt, out)