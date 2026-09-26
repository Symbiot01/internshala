"""Dense retrieval with multilingual-e5-small (MIT).

Encoding follows the model card: "query: " prefix on both sides for symmetric
similarity, mean pooling over tokens, L2 normalization. Search is exact:
a chunk of queries times all documents on the GPU, then top-k.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import config
import distributed

ENCODE_BATCH = 1024
QUERY_CHUNK = 1024
DOC_CHUNK = 400_000  # exact search in tiles so 12 GB and 80 GB cards both fit


def _from_pretrained(loader, path: str, **kwargs):
    """Use the local cache when present; download on a new machine otherwise."""
    try:
        return loader(path, local_files_only=True, **kwargs)
    except (OSError, ValueError):
        return loader(path, **kwargs)


def load_encoder(path: str = config.DENSE_MODEL, device: torch.device | str | None = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _from_pretrained(AutoTokenizer.from_pretrained, path)
    model = _from_pretrained(AutoModel.from_pretrained, path, dtype=torch.bfloat16)
    return tokenizer, model.to(device).eval()


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


@torch.inference_mode()
def encode(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device | str | None = None,
    rank: int = 0,
    world: int = 1,
    gather_dir: Path | None = None,
) -> np.ndarray:
    """Float16 unit vectors, one row per text, in input order. Ranks encode a stride and gather."""
    if device is None:
        device = next(model.parameters()).device
    dim = model.config.hidden_size
    indices = np.arange(len(texts))[rank::world]
    mine = [texts[i] for i in indices]
    order = np.argsort([len(item) for item in mine], kind="stable")
    part = np.zeros((len(mine), dim), dtype=np.float16)
    for start in range(0, len(mine), ENCODE_BATCH):
        rows = order[start : start + ENCODE_BATCH]
        batch = tokenizer([config.DENSE_PREFIX + mine[i] for i in rows], max_length=config.DENSE_MAX_LEN,
                          truncation=True, padding=True, return_tensors="pt").to(device)
        hidden = model(**batch).last_hidden_state
        vectors = torch.nn.functional.normalize(mean_pool(hidden, batch["attention_mask"]).float(), dim=-1)
        part[rows] = vectors.to(torch.float16).cpu().numpy()
        if start == 0 or (start // ENCODE_BATCH) % 50 == 0:
            print(f"dense encode rank {rank} {min(start + ENCODE_BATCH, len(mine))}/{len(mine)}", flush=True)
    if world == 1:
        return part
    if gather_dir is None:
        gather_dir = config.CACHE_DIR / "tmp_encode"
    gather_dir.mkdir(parents=True, exist_ok=True)
    np.save(gather_dir / f"part_{rank}.npy", part)
    distributed.barrier()
    out = np.zeros((len(texts), dim), dtype=np.float16)
    for r in range(world):
        out[r::world] = np.load(gather_dir / f"part_{r}.npy")
    return out


def _merge_topk(best_scores: torch.Tensor, best_index: torch.Tensor, scores: torch.Tensor, index: torch.Tensor, k: int):
    combined_scores = torch.cat([best_scores, scores], dim=1)
    combined_index = torch.cat([best_index, index], dim=1)
    values, order = torch.topk(combined_scores, k=k, dim=1)
    return values, torch.gather(combined_index, 1, order)


@torch.inference_mode()
def search(
    queries: np.ndarray,
    documents: np.ndarray,
    k: int,
    device: torch.device | str | None = None,
    rank: int = 0,
    world: int = 1,
    gather_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by cosine similarity. Documents are tiled so VRAM stays bounded."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = min(k, len(documents))
    q_index = np.arange(len(queries))[rank::world]
    q_slice = queries[q_index]
    positions_part = np.zeros((len(q_slice), k), dtype=np.int32)
    scores_part = np.zeros((len(q_slice), k), dtype=np.float32)
    for start in range(0, len(q_slice), QUERY_CHUNK):
        query = torch.from_numpy(q_slice[start : start + QUERY_CHUNK]).to(device)
        best_scores = torch.full((len(query), k), -1.0e4, device=device)
        best_index = torch.zeros((len(query), k), dtype=torch.long, device=device)
        for doc_start in range(0, len(documents), DOC_CHUNK):
            docs = torch.from_numpy(documents[doc_start : doc_start + DOC_CHUNK]).to(device)
            take = min(k, docs.shape[0])
            chunk_scores, chunk_index = torch.topk((query @ docs.T).float(), k=take, dim=1)
            chunk_index = chunk_index + doc_start
            if take < k:
                pad = k - take
                chunk_scores = torch.nn.functional.pad(chunk_scores, (0, pad), value=-1.0e4)
                chunk_index = torch.nn.functional.pad(chunk_index, (0, pad), value=0)
            best_scores, best_index = _merge_topk(best_scores, best_index, chunk_scores, chunk_index, k)
            del docs
        positions_part[start : start + len(query)] = best_index.int().cpu().numpy()
        scores_part[start : start + len(query)] = best_scores.cpu().numpy()
        if start == 0 or (start // QUERY_CHUNK) % 20 == 0:
            print(f"dense search rank {rank} {min(start + QUERY_CHUNK, len(q_slice))}/{len(q_slice)}", flush=True)
    torch.cuda.empty_cache()
    if world == 1:
        return positions_part, scores_part
    if gather_dir is None:
        gather_dir = config.CACHE_DIR / "tmp_search"
    gather_dir.mkdir(parents=True, exist_ok=True)
    np.savez(gather_dir / f"part_{rank}.npz", positions=positions_part, scores=scores_part)
    distributed.barrier()
    positions = np.zeros((len(queries), k), dtype=np.int32)
    scores = np.zeros((len(queries), k), dtype=np.float32)
    for r in range(world):
        payload = np.load(gather_dir / f"part_{r}.npz")
        positions[r::world] = payload["positions"]
        scores[r::world] = payload["scores"]
    return positions, scores
