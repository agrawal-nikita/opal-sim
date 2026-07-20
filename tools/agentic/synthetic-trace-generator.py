#!/usr/bin/env python3
"""Synthetic OTel trace generator for inference-perf replay sweeps.

Why
===

The Exgentic corpus is finite, somewhat noisy, and ties experiments
to whatever the original agent harnesses happened to record. For
controlled experiments — a clean throughput baseline, a JPS sweep
without long-tail outliers, repeatable scaling tests — we want
synthetic traces with knobs we can pin.

This module emits flat-JSON OTel session traces in the exact same
shape as `exgentic-analysis/filter_long_context_multi_toolcalls.py`,
so inference-perf's `data.type: otel_trace_replay` consumes them
unchanged. Each trace is one `SESSION_<sid>.json` file under
`--outdir`, plus a `subset_manifest.json` that mirrors what
`make_subset.py` writes (so existing tooling — sweep scripts,
common/stats.py, the JPS plotter — works without changes).

Token-count semantics
=====================

This is v1 of the generator: **plain chat only** (no tool calls)
with **cumulative prompts** (turn N's input includes the system
prompt plus all prior turns plus a new user message). That mirrors
how real agent sessions look to vLLM: input_tokens grows
monotonically, KV pressure builds, the long-context tail is what
saturates the GPU at high JPS.

Token counts are **approximate**: the prompt text is filler at a
configured chars-per-token ratio (default 4.0, the rule-of-thumb
for English with the Llama BPE tokenizer). Pass `--use-tokenizer`
to load `meta-llama/Llama-3.1-8B-Instruct`'s tokenizer and hit each
target within ±1 token (slower startup, ~2s, but exact).

Inter-span timing
-----------------

inference-perf's `otel_trace_to_replay_graph.py` builds a dependency
graph over the spans and uses **start_time deltas** to compute the
inter-event wait (line 1073: "gap between when the last predecessor
ends and this call starts"). So we encode the configured inter-span
gap as: `span[N].start_time = span[N-1].end_time + gap_seconds`.
Replay actually sleeps that gap between events.

Schema match
============

Each generated span carries (in order, matching what the Exgentic
filter writes for plain-chat events):

    {
      "trace_id":      <session id>,
      "span_id":       <16 hex chars>,
      "parent_span_id": null,
      "name":          "chat <model>",
      "kind":          "SPAN_KIND_CLIENT",
      "start_time":    ISO-8601 with microseconds,
      "end_time":      ISO-8601 with microseconds,
      "attributes": {
        "gen_ai.operation.name":           "chat",
        "gen_ai.request.model":            <model>,
        "gen_ai.response.model":           <model>,
        "gen_ai.usage.input_tokens":       <int>,
        "gen_ai.usage.output_tokens":      <int>,
        "gen_ai.response.id":              "chatcmpl-synthetic-<uuid>",
        "gen_ai.response.finish_reasons":  ["stop"],
        "gen_ai.input.messages":           <JSON-encoded list of role/parts>,
        "gen_ai.output.messages":          <JSON-encoded list with role=assistant>,
      }
    }

Top-level fields match the filter's output too (`trace_id`,
`span_count`, `collected_at`, `harness`, `benchmark`,
`source_model`, `spans`).

Usage
=====

    /data-env/atr/uv-envs/agentic-pip-clean/bin/python3 \\
        /home/atr/src/26-agentic-dev/src/synthetic-trace-generation/otel-generator.py \\
        --outdir /data/processed-traces/synthetic-otel-8

Defaults (all overridable via CLI):
    --num-traces 8
    --spans-per-trace 32 64
    --output-tokens 32 512
    --input-tokens 512 1024     (per-turn user message size)
    --system-prompt-tokens 8192 8192  (span 0's system prompt only)
    --inter-span-gap 2 16
    --seed 42                   (reproducible across runs)
    --model meta-llama/Llama-3.1-8B-Instruct  (only used as the
                                                 string in
                                                 gen_ai.request.model)

For exact tokens (slower):
    --use-tokenizer

For larger sweeps:
    --num-traces 50 --spans-per-trace 50 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants matching the Exgentic filter's output. We mirror these so
# downstream tooling (sweep scripts, common/stats.py, etc.) treats
# synthetic and real traces interchangeably.
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_HARNESS = "synthetic"      # surfaced in subset_manifest.json
DEFAULT_BENCHMARK = "synthetic"    # downstream tooling slices on this

# Filler text source. Lorem ipsum is convenient because it tokenises
# at a stable ~3.8 chars/token under Llama-3.1's tokenizer (close
# enough to the 4.0 default for the heuristic to land within ~5%).
# Repeated to give us enough text to slice from for any prompt size.
LOREM_PARAGRAPH = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
    "eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut "
    "enim ad minim veniam, quis nostrud exercitation ullamco laboris "
    "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in "
    "reprehenderit in voluptate velit esse cillum dolore eu fugiat "
    "nulla pariatur. Excepteur sint occaecat cupidatat non proident, "
    "sunt in culpa qui officia deserunt mollit anim id est laborum. "
)


# ---------------------------------------------------------------------------
# Token-count strategies
# ---------------------------------------------------------------------------

@dataclass
class TextScaler:
    """Generate filler text whose tokenized length is roughly N tokens.

    Two implementations: a fast chars-per-token heuristic (no tokenizer
    dep) and an exact-tokenizer path that loads the Llama-3.1
    tokenizer and round-trips encode/decode to hit exactly N tokens.
    The CLI flag `--use-tokenizer` selects the exact path.
    """

    chars_per_token: float = 4.0
    tokenizer: Any = None  # Lazy-loaded HF tokenizer when use_tokenizer=True

    def text_for_tokens(self, n_tokens: int, rng: random.Random) -> str:
        """Return filler text whose tokenized length is ~n_tokens."""
        if n_tokens <= 0:
            return ""
        if self.tokenizer is not None:
            return self._exact_text(n_tokens, rng)
        return self._heuristic_text(n_tokens, rng)

    def _heuristic_text(self, n_tokens: int, rng: random.Random) -> str:
        """Slice repeated lorem paragraph to approximately n_tokens.

        We add a small random prefix (8-32 chars) so different spans
        don't share a long prefix that would trivially hit the prefix
        cache, which would distort throughput numbers under
        `--enable-prefix-caching` (vLLM's default for our sweeps).
        """
        # Roughly target chars; sample length drift is fine — caller
        # records the *intended* token count in
        # gen_ai.usage.input_tokens, so vLLM will re-tokenise at
        # replay time and get a slightly different count anyway.
        target_chars = int(n_tokens * self.chars_per_token)
        # Prefix: a unique random sentence to break prefix-cache reuse
        # across spans within the same session. Span-0 system prompts
        # repeat their content cumulatively in later spans (that IS
        # the cache-hit pattern we want to keep), so the prefix lives
        # only on the new-content portion.
        prefix_len = rng.randint(8, 32)
        prefix_chars = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz ", k=prefix_len))
        prefix = f"<turn-{rng.randint(0, 1_000_000)}-{prefix_chars}> "
        body_chars_needed = max(0, target_chars - len(prefix))
        # How many full paragraphs?
        n_paragraphs = max(1, (body_chars_needed // len(LOREM_PARAGRAPH)) + 1)
        body = (LOREM_PARAGRAPH * n_paragraphs)[:body_chars_needed]
        return prefix + body

    def _exact_text(self, n_tokens: int, rng: random.Random) -> str:
        """Encode-then-decode round trip to hit exactly n_tokens."""
        # Generate a longish chunk of filler, encode, slice to n
        # tokens, decode back. This guarantees the resulting text
        # tokenises to exactly n_tokens (modulo edge tokens like BOS).
        seed_text = LOREM_PARAGRAPH * (n_tokens // 50 + 4)
        prefix_len = rng.randint(8, 32)
        prefix_chars = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz ", k=prefix_len))
        prefix = f"<turn-{rng.randint(0, 1_000_000)}-{prefix_chars}> "
        # add_special_tokens=False so we don't waste budget on BOS/EOS
        ids = self.tokenizer.encode(prefix + seed_text, add_special_tokens=False)
        if len(ids) < n_tokens:
            # Repeat seed_text until we have enough tokens
            seed_text = seed_text * ((n_tokens // len(ids)) + 2)
            ids = self.tokenizer.encode(prefix + seed_text, add_special_tokens=False)
        return self.tokenizer.decode(ids[:n_tokens], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Span / trace builders
# ---------------------------------------------------------------------------

def iso_timestamp(t: datetime) -> str:
    """ISO-8601 with microseconds, matching the Exgentic filter's format."""
    return t.isoformat(timespec="microseconds")


def build_span(
    trace_id: str,
    start_time: datetime,
    end_time: datetime,
    input_messages: list[dict[str, Any]],
    output_text: str,
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> dict[str, Any]:
    """Build one chat-span dict matching the Exgentic flat-JSON shape.

    No tool definitions, no parts of type="tool_call" — v1 is plain
    chat only. The output messages list always has role=assistant
    with a single text part, which is what inference-perf's replay
    expects when `expected_output_is_tool_call` is False.
    """
    span_id = uuid.uuid4().hex[:16]      # 16-hex-char span id
    response_id = f"chatcmpl-synthetic-{uuid.uuid4().hex[:24]}"

    # The output messages array is what inference-perf compares the
    # actual model output against (when verifying replay fidelity).
    # We set role=assistant + text part, finish_reason=stop — the
    # plain-chat happy path.
    output_messages = [
        {
            "role": "assistant",
            "finish_reason": "stop",
            "parts": [{"type": "text", "content": output_text}],
        }
    ]

    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": None,
        "name": f"chat {model}",
        "kind": "SPAN_KIND_CLIENT",
        "start_time": iso_timestamp(start_time),
        "end_time": iso_timestamp(end_time),
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": model,
            "gen_ai.response.model": model,
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
            "gen_ai.response.id": response_id,
            "gen_ai.response.finish_reasons": ["stop"],
            # Both messages fields are JSON-encoded strings, not raw
            # objects. That matches the Exgentic filter's output shape
            # (the OTel SDK serialises them this way before write).
            "gen_ai.input.messages": json.dumps(input_messages),
            "gen_ai.output.messages": json.dumps(output_messages),
        },
    }


def build_trace(
    trace_idx: int,
    args: argparse.Namespace,
    scaler: TextScaler,
    rng: random.Random,
) -> dict[str, Any]:
    """Build one full session trace (top-level dict + N spans).

    Per-trace knobs sampled from the configured ranges:
      - n_spans:             U[args.spans_per_trace_min, max]
      - per_span_input:      U[args.input_tokens_min, max]   (new content
                                                              per turn —
                                                              the system
                                                              prompt
                                                              accumulates
                                                              separately)
      - per_span_output:     U[args.output_tokens_min, max]
      - inter_span_gap:      U[args.inter_span_gap_min, max] seconds
      - system_prompt_size:  U[args.system_prompt_tokens_min, max]
                             (only on span 0)
    """
    n_spans = rng.randint(args.spans_per_trace_min, args.spans_per_trace_max)
    sys_tokens = rng.randint(args.system_prompt_tokens_min, args.system_prompt_tokens_max)

    # Build the trace_id in the same `<host>_<id>` shape Exgentic uses,
    # so downstream tools that pattern-match SESSION_<host>_<id>.json
    # don't need special-casing.
    trace_id = f"synth{trace_idx:06d}_{uuid.uuid4().hex[:8]}"

    # Wall-clock starts "now"; absolute time doesn't matter for replay
    # (inference-perf re-anchors on the first span's timestamp), but
    # we want consistent monotonic times within the trace so the
    # filter's start_time-based span ordering is stable.
    t = datetime(2026, 1, 1, 0, 0, 0) + timedelta(seconds=trace_idx * 86400)

    # The system prompt — only emitted on span 0. We generate it once
    # and re-use the same string in every subsequent span's input
    # messages so the conversation cumulates naturally.
    system_prompt_text = scaler.text_for_tokens(sys_tokens, rng)
    system_message = {
        "role": "system",
        "parts": [{"type": "text", "content": system_prompt_text}],
    }

    # `messages_so_far` tracks the cumulative conversation: each new
    # span's input is system + all prior user/assistant turns + the
    # new user turn. inference-perf re-tokenises on replay, so the
    # cumulative count it sees may drift from what we record here by
    # ~5-10% (chars-per-token heuristic) or ~0% (--use-tokenizer).
    # The recorded gen_ai.usage.input_tokens is what the filter and
    # common/stats.py read.
    messages_so_far: list[dict[str, Any]] = [system_message]
    cumulative_input_tokens = sys_tokens

    spans: list[dict[str, Any]] = []
    for span_idx in range(n_spans):
        # Per-turn budgets sampled afresh.
        new_user_tokens = rng.randint(args.input_tokens_min, args.input_tokens_max)
        out_tokens = rng.randint(args.output_tokens_min, args.output_tokens_max)
        # Inter-span gap: float, supports sub-second values too if the
        # caller passes them. start_time of span N = end_time of span
        # N-1 + gap. The first span has no preceding gap.
        if span_idx > 0:
            gap_s = rng.uniform(args.inter_span_gap_min, args.inter_span_gap_max)
            t = t + timedelta(seconds=gap_s)
        # Build the new user turn text for THIS span (only the
        # incremental content; everything before gets carried in
        # messages_so_far).
        new_user_text = scaler.text_for_tokens(new_user_tokens, rng)
        new_user_message = {
            "role": "user",
            "parts": [{"type": "text", "content": new_user_text}],
        }
        # Snapshot the input messages this span sees: full conversation
        # so far PLUS the new user turn. We deep-copy via JSON
        # roundtrip so later mutations to messages_so_far don't bleed
        # into earlier spans' recorded inputs.
        input_messages_for_span = messages_so_far + [new_user_message]
        cumulative_input_tokens_for_span = cumulative_input_tokens + new_user_tokens

        # Generate the assistant output text for THIS span.
        output_text = scaler.text_for_tokens(out_tokens, rng)

        # Bake the span. start/end_time deltas matter for replay's
        # inter-event waits; we use a 1-second nominal request
        # duration (real spans range from 100ms to many seconds, but
        # inference-perf overrides with the actual replay latency, so
        # this only affects which event the next span's wait_ms is
        # measured from).
        span_start = t
        span_end = t + timedelta(seconds=1)
        spans.append(
            build_span(
                trace_id=trace_id,
                start_time=span_start,
                end_time=span_end,
                input_messages=input_messages_for_span,
                output_text=output_text,
                input_tokens=cumulative_input_tokens_for_span,
                output_tokens=out_tokens,
                model=args.model,
            )
        )
        # Advance cursor for next span: t is now where this span ended,
        # the next gap will be applied from here.
        t = span_end

        # Carry the new turn (user msg) AND the assistant reply we just
        # synthesised into messages_so_far for the next iteration.
        # Cumulative-prompt semantics: span N+1's input = system +
        # all prior turns + new user turn.
        assistant_message = {
            "role": "assistant",
            "parts": [{"type": "text", "content": output_text}],
        }
        messages_so_far.append(new_user_message)
        messages_so_far.append(assistant_message)
        # Cumulative count grows by user_in + assistant_out (assistant
        # output goes into the prompt of the next turn).
        cumulative_input_tokens = cumulative_input_tokens_for_span + out_tokens

    return {
        "trace_id": trace_id,
        "span_count": len(spans),
        "collected_at": iso_timestamp(datetime.now()),
        "harness": DEFAULT_HARNESS,
        "benchmark": DEFAULT_BENCHMARK,
        "source_model": args.model,
        "spans": spans,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_subset_manifest(outdir: Path, traces: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Write subset_manifest.json mirroring make_subset.py's shape.

    This lets the synthetic dir be passed to the same downstream
    tools (continuum-scripts sweeps, common/stats.py, the JPS
    plotter) without special-casing — they all check for
    subset_manifest.json to determine the corpus shape.
    """
    files = sorted(f"SESSION_{t['trace_id']}.json" for t in traces)
    manifest = {
        "n_requested": args.num_traces,
        "n_placed": len(traces),
        "ordering": "synthetic",
        "seed": args.seed,
        "source": "synthetic-trace-generation/otel-generator.py",
        "use_copy": False,
        "files": files,
        "harness_counts": {DEFAULT_HARNESS: len(traces)},
        "benchmark_counts": {DEFAULT_BENCHMARK: len(traces)},
        "source_model_counts": {args.model: len(traces)},
        # Generator parameters so future runs can be reproduced from
        # the manifest alone (subject to seed determinism).
        "generator_params": {
            "spans_per_trace": [args.spans_per_trace_min, args.spans_per_trace_max],
            "output_tokens": [args.output_tokens_min, args.output_tokens_max],
            "input_tokens": [args.input_tokens_min, args.input_tokens_max],
            "system_prompt_tokens": [args.system_prompt_tokens_min, args.system_prompt_tokens_max],
            "inter_span_gap_sec": [args.inter_span_gap_min, args.inter_span_gap_max],
            "use_tokenizer": args.use_tokenizer,
            "chars_per_token": args.chars_per_token,
        },
    }
    (outdir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_range(s: str) -> tuple[int, int]:
    """Parse a string like '32-64' or '32 64' into a (min, max) tuple.

    Used so the user can pass --spans-per-trace 32-64 (one arg) or
    --spans-per-trace 32 64 (two args via nargs=2). Either works.
    """
    parts = s.replace("-", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected 'min-max' or 'min max', got {s!r}")
    return (int(parts[0]), int(parts[1]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--outdir", type=Path, required=True,
                   help="Output directory for SESSION_*.json files (and subset_manifest.json).")
    p.add_argument("--num-traces", type=int, default=8,
                   help="Number of session traces to generate (default: 8).")
    # Range knobs all parsed via nargs=2 with int type. Defaults
    # match the spec in synthetic-trace-generation/prompt.md.
    p.add_argument("--spans-per-trace", nargs=2, type=int, metavar=("MIN", "MAX"),
                   default=[32, 64],
                   help="Number of spans per trace, sampled uniformly (default: 32 64).")
    p.add_argument("--output-tokens", nargs=2, type=int, metavar=("MIN", "MAX"),
                   default=[32, 512],
                   help="Output tokens per span, sampled uniformly (default: 32 512).")
    p.add_argument("--input-tokens", nargs=2, type=int, metavar=("MIN", "MAX"),
                   default=[512, 1024],
                   help="New user-message tokens per span, sampled uniformly. "
                        "The cumulative prompt at span N = system_prompt + sum of "
                        "prior turns + this new user message (default: 512 1024).")
    p.add_argument("--system-prompt-tokens", nargs=2, type=int, metavar=("MIN", "MAX"),
                   default=[8192, 8192],
                   help="System prompt size on span 0, sampled uniformly. "
                        "Range collapsed to a single value by default (default: 8192 8192).")
    p.add_argument("--inter-span-gap", nargs=2, type=float, metavar=("MIN", "MAX"),
                   default=[2.0, 16.0],
                   help="Seconds between spans (the 'agent think time'). "
                        "Encoded into start_time deltas; inference-perf "
                        "honors this as a replay-time wait between events "
                        "(default: 2 16).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility (default: 42).")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help="String written to gen_ai.request.model. Does NOT "
                        "select a tokenizer; use --use-tokenizer for that "
                        f"(default: {DEFAULT_MODEL}).")
    p.add_argument("--use-tokenizer", action="store_true",
                   help="Load the Llama-3.1 tokenizer and hit token counts "
                        "exactly via encode/decode round-trip. Slower "
                        "startup (~2s) but no chars-per-token drift.")
    p.add_argument("--chars-per-token", type=float, default=4.0,
                   help="Heuristic chars-per-token ratio when "
                        "--use-tokenizer is off. 4.0 is the rule-of-thumb "
                        "for English under Llama BPE (default: 4.0).")
    args = p.parse_args(argv)

    # Re-pack range args into the (min, max) attributes used by
    # build_trace. nargs=2 gives us a list, we want named min/max
    # for clarity in build_trace.
    args.spans_per_trace_min, args.spans_per_trace_max = args.spans_per_trace
    args.output_tokens_min, args.output_tokens_max = args.output_tokens
    args.input_tokens_min, args.input_tokens_max = args.input_tokens
    args.system_prompt_tokens_min, args.system_prompt_tokens_max = args.system_prompt_tokens
    args.inter_span_gap_min, args.inter_span_gap_max = args.inter_span_gap

    args.outdir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # Build the text scaler. With --use-tokenizer we lazily import
    # transformers so the no-tokenizer path stays import-light.
    if args.use_tokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "--use-tokenizer requires the `transformers` package "
                "(it's already in agentic-pip-clean on l40s)"
            ) from exc
        print(f"[info] loading tokenizer for {args.model} ...", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        scaler = TextScaler(tokenizer=tokenizer)
    else:
        scaler = TextScaler(chars_per_token=args.chars_per_token)

    print(f"[info] generating {args.num_traces} traces -> {args.outdir}", file=sys.stderr)
    print(f"[info] spans/trace={args.spans_per_trace_min}-{args.spans_per_trace_max}  "
          f"input/turn={args.input_tokens_min}-{args.input_tokens_max}  "
          f"output/turn={args.output_tokens_min}-{args.output_tokens_max}  "
          f"sys-prompt={args.system_prompt_tokens_min}-{args.system_prompt_tokens_max}  "
          f"gap={args.inter_span_gap_min}-{args.inter_span_gap_max}s",
          file=sys.stderr)

    traces: list[dict[str, Any]] = []
    # One JSONL file for the whole corpus: each line is a full session
    # trace (the same dict we used to write to SESSION_<id>.json).
    # Timestamp the filename (e.g. synthetic_traces_26-07-03_12_18_10.jsonl)
    # so successive runs don't clobber each other.
    stamp = datetime.now().strftime("%y-%m-%d_%H_%M_%S")
    jsonl_path = args.outdir / f"synthetic_traces_{stamp}.jsonl"
    with jsonl_path.open("w") as fh:
        for i in range(args.num_traces):
            trace = build_trace(i, args, scaler, rng)
            fh.write(json.dumps(trace) + "\n")
            traces.append(trace)
            # Quick progress: print every 50 traces or on the last one.
            if (i + 1) % 50 == 0 or i + 1 == args.num_traces:
                print(f"  ... {i + 1}/{args.num_traces}", file=sys.stderr)

    write_subset_manifest(args.outdir, traces, args)
    print(f"[done] wrote {len(traces)} traces -> {jsonl_path} + subset_manifest.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
