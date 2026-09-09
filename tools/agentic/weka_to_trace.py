# SPDX-License-Identifier: Apache-2.0
"""Reformat a Weka agentic-trace export (raw .jsonl, or the HuggingFace
auto-converted .parquet mirror) into the JSONL format
opal.workloads.weka_trace.WekaTrace replays.

Weka traces (e.g. semianalysisai/cc-traces-weka-no-subagents-051226) store,
per trace: {id, models, block_size, hash_id_scope, requests: [{t, type,
model, in, out, hash_ids, api_time, think_time, ttft}, ...]}. `hash_ids` is
one COMPACT hash per `block_size`-token block (64 tokens in that dataset) --
across the real dataset that's ~24 billion prompt tokens compressed into
~379 million compact block-hash entries. An identical hash value means
identical block content globally across the dataset (e.g. a shared system
prompt/tool-definition block reused by many independent sessions).

This script does NOT expand those block hashes into per-token ids -- doing
so here would blow the ~379M compact entries out into ~24 billion individual
JSON integers (100+ GB), which is neither writable nor loadable. Instead it
is a thin reformat (renaming fields, dropping ones the replay doesn't use)
that keeps `hash_ids` compact; opal.workloads.weka_trace.WekaTrace expands
each turn's compact hashes into real per-token ids lazily, at replay time,
with a cache -- exactly like opal.workloads.workload.Trace._expand_prompt
already does for the legacy `trace` workload type.

Usage:
    python tools/agentic/weka_to_trace.py /path/to/traces.jsonl
    python tools/agentic/weka_to_trace.py /path/to/weka.parquet [more.parquet ...]

Writes a sibling "-converted.jsonl" next to the first input file.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("weka_to_trace")


def convert_trace(row: dict) -> dict | None:
    """Convert one Weka trace row into a {trace_id, requests: [...]} session.

    Keeps `hash_ids` compact (one hash per block_size-token block) -- no
    expansion happens here, see the module docstring.
    """
    trace_id = row.get("id")
    requests_in = row.get("requests") or []
    if not requests_in:
        return None

    requests_out = []
    for turn in requests_in:
        input_length = int(turn["in"])
        hash_ids = turn.get("hash_ids") or []
        if not hash_ids and input_length > 0:
            log.warning(f"trace {trace_id}: turn has in={input_length} but no hash_ids -- skipping turn")
            continue
        requests_out.append({
            "t": float(turn["t"]),
            "think_time": float(turn.get("think_time") or 0.0),
            "input_length": input_length,
            "hash_ids": hash_ids,
            "output_length": int(turn.get("out") or 1),
        })

    if not requests_out:
        return None
    return {"trace_id": trace_id, "requests": requests_out}


def _iter_rows_jsonl(path: Path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_rows_parquet(path: Path):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    yield from table.to_pylist()


def iter_rows(input_files: list[Path]):
    for path in input_files:
        if path.suffix == ".parquet":
            yield from _iter_rows_parquet(path)
        else:
            yield from _iter_rows_jsonl(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input_files", type=Path, nargs="+", help="one or more local Weka traces.jsonl or parquet shard(s)"
    )
    parser.add_argument("--limit", type=int, default=0, help="stop after N traces (0 = all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    first = args.input_files[0]
    out_path = first.with_name(f"{first.stem}-converted.jsonl")
    if out_path.exists():
        parser.error(f"{out_path} already exists; remove it first")

    written = skipped = turns = 0
    with open(out_path, "w") as fout:
        for row in iter_rows(args.input_files):
            session = convert_trace(row)
            if session is None:
                skipped += 1
                continue
            fout.write(json.dumps(session) + "\n")
            written += 1
            turns += len(session["requests"])
            if written % 100 == 0:
                log.info(f"converted {written} trace(s), {turns} turn(s) so far...")
            if args.limit and written >= args.limit:
                break

    log.info(f"wrote {written} session(s), {turns} turn(s) -> {out_path}; skipped {skipped} trace(s)")


if __name__ == "__main__":
    main()
