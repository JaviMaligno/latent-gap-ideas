"""Local Qwen3-Embedding-0.6B (ONNX, CPU): same vectors as sentence-transformers on Kaggle (last-token pooling,
L2 norm, 512 tokens, documents embedded without an instruction prompt)."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

MODEL_DIR = Path.home() / "Documents/experiment-data/latent-gap-ideas/models/qwen3e06-onnx"
MAX_TOKENS = 512
EOS_ID = 151643  # <|endoftext|>


class LocalQwen:
    def __init__(self, model_dir: Path = MODEL_DIR, threads: int = 8):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model_dir / "onnx/model.onnx"), opts, providers=["CPUExecutionProvider"])
        self.kv = [i.name for i in self.sess.get_inputs() if i.name.startswith("past_key_values")]
        self._lock = threading.Lock()  # one ONNX run at a time; it already uses every core

    def _ids(self, text: str) -> list[int]:
        ids = self.tok.encode(text).ids
        if not ids or ids[-1] != EOS_ID:
            ids = ids + [EOS_ID]
        if len(ids) > MAX_TOKENS:
            ids = ids[:MAX_TOKENS - 1] + [EOS_ID]
        return ids

    def embed_one(self, text: str) -> np.ndarray:
        ids = np.array([self._ids(text)], dtype=np.int64)
        n = ids.shape[1]
        feed = {"input_ids": ids, "attention_mask": np.ones((1, n), dtype=np.int64),
                "position_ids": np.arange(n, dtype=np.int64)[None]}
        empty = np.zeros((1, 8, 0, 128), dtype=np.float32)
        feed.update({k: empty for k in self.kv})
        with self._lock:
            h = self.sess.run(["last_hidden_state"], feed)[0][0, -1]
        return (h / np.linalg.norm(h)).astype(np.float32)

    def embed(self, texts: list[str], tag: str = "") -> np.ndarray:
        return np.stack([self.embed_one(t) for t in texts])
