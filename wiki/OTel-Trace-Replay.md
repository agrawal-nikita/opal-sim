## OTel Trace Replay

Opal can replay real agentic traffic captured as [OpenTelemetry](https://opentelemetry.io/) `gen_ai` traces. Each session's turns are re-submitted to the simulated cluster at their recorded wall-clock timing, so you can characterize KV-cache behaviour, latency, and throughput against production-shaped workloads.

OTel replay is its own workload type (`otel`), separate from the legacy `trace` workload. Pick the right one:

| Your trace file looks like… | Use type |
|---|---|
| Flat JSONL rows: `{"timestamp", "input_length", "output_length", "hash_ids"}` | `trace` (see [[Running Workloads]]) |
| OTel sessions/spans: `{"trace_id", "spans": [...]}` or pre-tokenized `{"trace_id", "turns": [...]}` | `otel` (this page) |

> They can share the `.jsonl` extension, so the distinction is the **content**, not the filename. Setting the wrong `type` produces confusing errors (e.g. `AssertionError: 0 != None` when an OTel file is fed to the legacy `trace` workload).

### Minimal config

Add a stage with `"type": "otel"` under `workload.stages`. The two required params are `trace_file` and (for raw traces) `tokenizer`:

```json
"workload": {
  "stages": [
    {
      "type": "otel",
      "workload_params": {
        "trace_file": "traces/exagentic_v2_swebench.jsonl",
        "tokenizer": "meta-llama/Llama-3.1-8B-Instruct",
        "pretokenized": false,
        "total_requests": 10,
        "multiplier_to_sec": 1,
        "inter_turn_multiplier": 1,
        "max_concurrent_sessions": 2
      }
    }
  ]
}
```

### What must change vs. a legacy `trace` stage

| Change | From (`trace`) | To (`otel`) |
|---|---|---|
| Stage type | `"type": "trace"` | `"type": "otel"` |
| Tokenizer | not used | `"tokenizer"` **required** unless `pretokenized: true` |
| Pre-tokenized flag | n/a | `"pretokenized"` selects the raw vs. tokenized loader |
| `chunk_size` | used for hash expansion | **remove it** — OTel replay carries real token ids, no expansion |
| Session concurrency | n/a | `"max_concurrent_sessions"` controls parallelism |

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `trace_file` | yes | — | Path to the OTel `.json`/`.jsonl` trace. |
| `tokenizer` | yes, if not pre-tokenized | — | HuggingFace tokenizer name (e.g. `meta-llama/Llama-3.1-8B-Instruct`). Raw traces are tokenized on the fly, once per turn per run. |
| `pretokenized` | no | `false` | `true` for traces already tokenized by `tools/agentic/tokenizer.py` (they carry token ids; **no** tokenizer is loaded and `tokenizer` is ignored). |
| `total_requests` | no | `-1` (all) | Multi-session: cap on the number of **sessions** started. Single-session: cap on the number of **spans** replayed. |
| `multiplier_to_sec` | no | `1` | Scales trace timestamps → simulation seconds (also scales session arrival spacing). e.g. `0.001` if timestamps are in ms; `<1` to fast-forward. |
| `inter_turn_multiplier` | no | `1` | Multi-session only. Scales **only** the gap between a turn finishing and the next turn's wall start. `0` replays a session's turns back-to-back. |
| `max_concurrent_sessions` | no | `10` | Multi-session only. Max sessions replaying at once (FIFO semaphore). `-1` starts all sessions concurrently. |

Single- vs. multi-session mode is detected automatically from the trace. In multi-session mode, turns **within** a session are replayed strictly in order — each turn is submitted only after the previous one completes — while up to `max_concurrent_sessions` sessions run in parallel.

### Pre-tokenizing a trace (recommended)

Raw traces are re-tokenized every run, which dominates startup for large traces. Tokenize once instead:

```bash
python tools/agentic/tokenizer.py traces/exagentic_v2_swebench.jsonl \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

This writes `traces/exagentic_v2_swebench-tokenized.jsonl`. Point `trace_file` at it and set `pretokenized: true` (the `tokenizer` param is then unused). Same simulation, far faster startup:

```json
{
  "type": "otel",
  "workload_params": {
    "trace_file": "traces/exagentic_v2_swebench-tokenized.jsonl",
    "pretokenized": true,
    "max_concurrent_sessions": 2
  }
}
```

### Running

Point `main.py` at a config whose workload stage is the `otel` stage above:

```bash
python opal/main.py -c configs/defaults.json -o out/
```

### Prerequisites

- `opal/workloads/otel_tokenizer.py` must be present (provides the `OtelTokenizer` used to load and tokenize OTel traces).
- The `transformers` and `jinja2` packages must be installed (they are listed in `pyproject.toml`; `uv pip install -e .` pulls them in).
