"""Small JSONL and Responses API helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_checkpoint(path: Path) -> list[dict]:
    """Read a checkpoint, repairing a final line interrupted during a write."""
    lines = path.read_text().splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if any(rest.strip() for rest in lines[index + 1:]):
                raise
            write_jsonl(path, records)
            break
    return records


def response_output_text(body: dict) -> str:
    pieces = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                pieces.append(content.get("text", ""))
    if not pieces:
        raise ValueError("Response contained no output_text")
    return "".join(pieces)


def successful_batch_bodies(path: Path) -> dict[str, dict]:
    result = {}
    for line in read_jsonl(path):
        custom_id = line["custom_id"]
        response = line.get("response")
        if line.get("error") or not response or response.get("status_code") != 200:
            raise ValueError(f"Batch request failed for {custom_id}: {line.get('error') or response}")
        result[custom_id] = response["body"]
    return result
