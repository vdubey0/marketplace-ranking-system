"""Prepare, submit, inspect, and download OpenAI Batch API jobs."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

ACQUISITION_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ACQUISITION_ROOT.parent


def load_api_key() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to the project-root .env file.")


def read_state(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def write_state(path: Path, batch, **extra) -> dict:
    state = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch.id,
        "status": batch.status,
        "input_file_id": batch.input_file_id,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "request_counts": batch.request_counts.model_dump() if batch.request_counts else None,
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")
    return state


def submit(input_path: Path, state_path: Path) -> dict:
    load_api_key()
    client = OpenAI()
    with input_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"pipeline": "synthetic-resumes", "input": input_path.name},
    )
    return write_state(state_path, batch, input_path=str(input_path.resolve()))


def refresh(state_path: Path) -> dict:
    load_api_key()
    previous = read_state(state_path)
    batch = OpenAI().batches.retrieve(previous["batch_id"])
    return write_state(state_path, batch, input_path=previous.get("input_path"))


def wait_for_completion(state_path: Path, poll_seconds: float) -> dict:
    if poll_seconds < 2:
        raise ValueError("poll-seconds must be at least 2")
    state = refresh(state_path)
    counts = state.get("request_counts") or {}
    total = int(counts.get("total", 0))
    completed = int(counts.get("completed", 0)) + int(counts.get("failed", 0))
    progress = tqdm(total=total or None, initial=completed, desc="OpenAI batch",
                    unit="request", dynamic_ncols=True)
    terminal = {"completed", "failed", "expired", "cancelled"}
    try:
        while state["status"] not in terminal:
            progress.set_postfix(status=state["status"])
            time.sleep(poll_seconds)
            state = refresh(state_path)
            counts = state.get("request_counts") or {}
            new_total = int(counts.get("total", 0))
            if new_total:
                progress.total = new_total
            done = int(counts.get("completed", 0)) + int(counts.get("failed", 0))
            progress.update(max(0, done - progress.n))
            progress.refresh()
    finally:
        progress.close()
    if state["status"] != "completed":
        raise RuntimeError(f"Batch ended with status {state['status']}")
    return state


def download(state_path: Path, output_path: Path, error_path: Path | None) -> dict:
    load_api_key()
    state = refresh(state_path)
    if state["status"] != "completed":
        raise RuntimeError(f"Batch is {state['status']}; results are not ready")
    client = OpenAI()
    if state["output_file_id"]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(client.files.content(state["output_file_id"]).text)
    if state["error_file_id"] and error_path:
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(client.files.content(state["error_file_id"]).text)
    state["downloaded_output"] = str(output_path.resolve())
    if error_path and error_path.exists():
        state["downloaded_errors"] = str(error_path.resolve())
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--input", type=Path, required=True)
    submit_parser.add_argument("--state", type=Path, required=True)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--state", type=Path, required=True)
    wait_parser = sub.add_parser("wait")
    wait_parser.add_argument("--state", type=Path, required=True)
    wait_parser.add_argument("--poll-seconds", type=float, default=15)
    download_parser = sub.add_parser("download")
    download_parser.add_argument("--state", type=Path, required=True)
    download_parser.add_argument("--output", type=Path, required=True)
    download_parser.add_argument("--errors", type=Path)
    args = parser.parse_args()
    if args.command == "submit":
        result = submit(args.input, args.state)
    elif args.command == "status":
        result = refresh(args.state)
    elif args.command == "wait":
        result = wait_for_completion(args.state, args.poll_seconds)
    else:
        result = download(args.state, args.output, args.errors)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
