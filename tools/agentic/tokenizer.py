# SPDX-License-Identifier: Apache-2.0
"""Pre-tokenize an OTel trace so replay does not have to run a tokenizer.

Reads a raw trace (one session per line for .jsonl, or a single object / list of
objects for .json) and writes a sibling "-tokenized.jsonl" file, e.g.
traces/deepseek.jsonl -> traces/deepseek-tokenized.jsonl. The output is always
JSONL regardless of the input extension, because that is what the content is:
one session per line.

Each output line is one session, holding the fields an LLMRequest is built from:

    {"trace_id": ..., "turns": [{"span_id": ..., "start_time": ..., "end_time": ...,
                                 "input_length": N, "output_length": M,
                                 "hash_ids": [...], "output_token_ids": [...]}, ...]}

so a replay can do, per turn:

    request = LLMRequest(env, stage_id, turn["input_length"],
                         hash_ids=turn["hash_ids"], output_length=turn["output_length"])
    request.output_token_ids = turn["output_token_ids"]
    request.session_id = session["trace_id"]

start_time/end_time are carried through unchanged because the replay needs them
to schedule turns (the first turn relative to the session base, later turns
relative to the previous turn's end).

Usage:
    python tools/agentic/tokenizer.py traces/deepseek.jsonl --tokenizer meta-llama/Llama-3.1-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from opal.workloads.otel_tokenizer import OtelTokenizer

log = logging.getLogger("tokenizer")


def iter_sessions(path: Path):
    """Yield raw session objects one at a time (traces run to multiple GB)."""
    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        with open(path) as f:
            data = json.load(f)
        yield from (data if isinstance(data, list) else [data])


def tokenize_span(tokenizer: OtelTokenizer, span: dict) -> dict:
    """Tokenize one span into the fields an LLMRequest is built from.

    Mirrors the extraction the Trace workload does inline in
    opal/workloads/workload.py, and emits the keys that Trace._turn_fields()
    reads back, so a pre-tokenized turn is a drop-in for a raw span there.
    op_name/model/prompt_tokens are carried through for the replay logs only.
    """
    attrs = span["attributes"]
    messages = json.loads(attrs["gen_ai.input.messages"])
    raw_tool_defs = attrs.get("gen_ai.tool.definitions")
    tool_defs = json.loads(raw_tool_defs) if isinstance(raw_tool_defs, str) else raw_tool_defs
    system_instructions = attrs.get("gen_ai.system_instructions")
    token_ids = tokenizer.tokenize_messages(
        messages, tool_defs=tool_defs, system_instructions=system_instructions
    )

    raw_output = attrs.get("gen_ai.output.messages")
    output_token_ids = None
    if raw_output:
        output_token_ids = tokenizer.tokenize_output(
            messages, json.loads(raw_output), system_instructions=system_instructions
        )

    # Prefer the actual re-tokenized output length; fall back to the
    # trace-reported completion_tokens only when we have no output messages.
    output_length = (
        len(output_token_ids) if output_token_ids else int(attrs.get("gen_ai.usage.completion_tokens", 1))
    )

    return {
        "span_id": span.get("span_id"),
        "parent_span_id": span.get("parent_span_id"),
        "start_time": span.get("start_time"),
        "end_time": span.get("end_time"),
        "input_length": len(token_ids),
        "output_length": output_length,
        "op_name": attrs.get("gen_ai.operation.name", span.get("name", "?")),
        "model": attrs.get("gen_ai.request.model", attrs.get("gen_ai.response.model", "?")),
        "prompt_tokens": int(attrs.get("gen_ai.usage.prompt_tokens", 0)),
        "hash_ids": token_ids,
        "output_token_ids": output_token_ids,
    }


def tokenize_session(tokenizer: OtelTokenizer, session: dict) -> dict | None:
    """Tokenize every replayable span in a session, or None if it has none."""
    spans = [s for s in session.get("spans", []) if "gen_ai.input.messages" in s.get("attributes", {})]
    if not spans:
        return None
    spans.sort(key=lambda s: s["start_time"])
    return {
        "trace_id": spans[0].get("trace_id", session.get("trace_id")),
        "turns": [tokenize_span(tokenizer, s) for s in spans],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_file", type=Path, help="raw OTel trace, e.g. traces/deepseek.jsonl")
    parser.add_argument(
        "--tokenizer",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace tokenizer name (must match the model the sim replays as)",
    )
    parser.add_argument("--limit", type=int, default=0, help="stop after N sessions (0 = all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    # always .jsonl: the output is one session per line whatever the input was
    out_path = args.trace_file.with_name(f"{args.trace_file.stem}-tokenized.jsonl")
    if out_path.exists():
        parser.error(f"{out_path} already exists; remove it first")

    tokenizer = OtelTokenizer(args.tokenizer)
    started = time.monotonic()
    written = skipped = turns = 0

    with open(out_path, "w") as fout:
        for session in iter_sessions(args.trace_file):
            result = tokenize_session(tokenizer, session)
            if result is None:
                skipped += 1
                continue
            fout.write(json.dumps(result) + "\n")
            written += 1
            turns += len(result["turns"])
            log.info(
                f"session {written} trace_id={result['trace_id']} turns={len(result['turns'])} "
                f"input_tokens={sum(t['input_length'] for t in result['turns'])} "
                f"elapsed={time.monotonic() - started:.1f}s"
            )
            if args.limit and written >= args.limit:
                break

    log.info(
        f"wrote {written} session(s), {turns} turns -> {out_path} in {time.monotonic() - started:.1f}s"
        f"; skipped {skipped} session(s) with no replayable spans"
    )


if __name__ == "__main__":
    main()
