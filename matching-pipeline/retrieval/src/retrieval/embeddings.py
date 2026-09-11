"""Versioned, resumable embedding artifacts."""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_records(
    records: Sequence[tuple[str, str]],
    output_path: Path,
    embed_batch: Callable[[list[str]], Sequence[Sequence[float]]],
    *,
    model: str,
    dimensions: int,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Embed changed records and checkpoint after every successful API batch."""
    if dimensions <= 0 or batch_size <= 0:
        raise ValueError("dimensions and batch_size must be positive")
    wanted = [(str(entity_id), text, content_hash(text)) for entity_id, text in records]
    if len({entity_id for entity_id, _, _ in wanted}) != len(wanted):
        raise ValueError("entity IDs must be unique")

    columns = ["entity_id", "content_sha256", "model", "dimensions", "embedding"]
    checkpoint_dir = output_path.with_suffix(".parts")
    cached_frames = []
    if output_path.exists():
        cached_frames.append(pd.read_parquet(output_path))
    if checkpoint_dir.exists():
        cached_frames.extend(pd.read_parquet(path) for path in sorted(checkpoint_dir.glob("*.parquet")))
    cached = (pd.concat(cached_frames, ignore_index=True).drop_duplicates(
        ["entity_id", "content_sha256", "model", "dimensions"], keep="last"
    ) if cached_frames else pd.DataFrame(columns=columns))
    if list(cached.columns) != columns:
        raise ValueError(f"Unexpected embedding artifact schema: {output_path}")

    reusable = {
        (row.entity_id, row.content_sha256, row.model, int(row.dimensions)): row.embedding
        for row in cached.itertuples(index=False)
    }
    rows = []
    pending = []
    for entity_id, text, digest in wanted:
        key = (entity_id, digest, model, dimensions)
        if key in reusable:
            rows.append({"entity_id": entity_id, "content_sha256": digest, "model": model,
                         "dimensions": dimensions, "embedding": reusable[key]})
        else:
            pending.append((entity_id, text, digest))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        vectors = embed_batch([text for _, text, _ in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding provider returned the wrong number of vectors")
        batch_rows = []
        for (entity_id, _, digest), vector in zip(batch, vectors, strict=True):
            array = np.asarray(vector, dtype=np.float32)
            if array.shape != (dimensions,) or not np.isfinite(array).all():
                raise ValueError(f"Invalid embedding for {entity_id}")
            row = {"entity_id": entity_id, "content_sha256": digest, "model": model,
                   "dimensions": dimensions, "embedding": array.tolist()}
            rows.append(row)
            batch_rows.append(row)
        checkpoint_key = hashlib.sha256(
            "|".join(row["entity_id"] for row in batch_rows).encode("utf-8")
        ).hexdigest()[:16]
        pd.DataFrame(batch_rows, columns=columns).to_parquet(
            checkpoint_dir / f"part-{checkpoint_key}.parquet", index=False
        )

    result = pd.DataFrame(rows, columns=columns).sort_values("entity_id").reset_index(drop=True)
    result.to_parquet(output_path, index=False)
    return result


def embedding_matrix(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        raise ValueError("embedding artifact is empty")
    return np.stack(frame.embedding.map(lambda x: np.asarray(x, dtype=np.float32)))
