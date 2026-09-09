# SPDX-License-Identifier: Apache-2.0
"""Replay pre-converted Weka agentic traces (config: type "wekatrace").

Weka traces are session-shaped like OTel multi-session replay, but they carry
relative-second timestamps and COMPACT block hashes instead of ISO wall-clock
spans and gen_ai.* attributes -- see tools/agentic/weka_to_trace.py, which
reformats the raw Weka export (one hash per block_size-token block; the real
dataset compresses ~24 billion prompt tokens into ~379 million such compact
entries) into the flat JSONL this class reads.

Expanding every compact hash into a full per-token id list at conversion
time would blow that ~379M entries out into ~24 billion individual ids --
not writable or loadable. So the converted file keeps hash_ids compact, and
this class expands them into real per-token ids lazily, per turn, right
before submitting a request -- exactly like opal.workloads.workload.Trace
._expand_prompt does for the legacy `trace` workload. Two sessions that
share a block hash (e.g. a common system prompt/tool-definition block) get
identical expanded ids, so real cross-session prefix-cache sharing survives
the expansion; the result then flows through the normal vLLM/KVC block-
hashing pipeline like any other trace's real token ids.

Expected file: JSONL, one session per line:
    {"trace_id": ..., "requests": [{"t": <sec since session start>,
                                     "think_time": <sec gap since previous request's end>,
                                     "input_length": N, "hash_ids": [...compact block hashes...],
                                     "output_length": M}, ...]}

Turns within a session are replayed strictly in order -- each turn is
submitted only after the previous one completes -- while up to
max_concurrent_sessions sessions replay in parallel. Weka's "no-subagents"
traces already contain only sequential main-agent turns, so there is nothing
to parallelize within a session.
"""
from __future__ import annotations

import json
import logging
import math
import os

import simpy

from opal.core.request import LLMRequest
from opal.utils.util import safe_process
from opal.workloads.abstract_workload import AbstractWorkload


class WekaTrace(AbstractWorkload):
    def __init__(self, opal_env: "OpalSimulatorEnvironment", stage_id: int, workload_params: dict, req_router):
        super().__init__(opal_env, stage_id, workload_params, req_router, name="WekaTrace")
        params = self.workload_params["workload_params"]
        self.trace_file = params["trace_file"]
        # Multi-session cap on the number of sessions started (-1 = all), matching Otel's semantics.
        self.total_requests = int(params.get("total_requests", -1))
        self.multiplier_to_sec = float(params.get("multiplier_to_sec", 1))
        self.inter_turn_multiplier = float(params.get("inter_turn_multiplier", 1))
        self._max_concurrent_sessions = int(params.get("max_concurrent_sessions", 10))

        # Block-hash expansion, identical in approach to Trace._expand_prompt:
        # each compact hash covers block_size tokens (64 for the Weka corpus).
        self.block_size = int(params.get("block_size", 64))
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        self._hash_id_offset_bits = max(1, math.ceil(math.log2(max(2, self.block_size))))
        self._hash_id_base_bits = 31 - self._hash_id_offset_bits
        if self._hash_id_base_bits < 1:
            raise ValueError(
                f"block_size={self.block_size} too large for int32 hash_ids "
                f"(needs {self._hash_id_offset_bits} bits for offset, leaving {self._hash_id_base_bits} for base)"
            )
        self._hash_id_base_mask = (1 << self._hash_id_base_bits) - 1
        # Shared across the whole run (not per-session): a hash value that recurs in a
        # different session deterministically expands to the same synthetic ids, so real
        # cross-session prefix sharing (e.g. a common system prompt) is preserved.
        self._expanded_generated_prompts: dict = {}

        self.name = f"Workload(wekatrace {os.path.basename(self.trace_file)})"
        self.log = logging.getLogger(self.name)

        # Stop reading the file once we have enough sessions for total_requests, rather
        # than eagerly parsing/holding all 949+ sessions when only a subset is replayed.
        session_limit = self.total_requests if self.total_requests >= 0 else -1
        self._sessions = self._load_sessions(self.trace_file, session_limit)

        self.request_id = 0
        self.is_finished = False
        self.log.info(
            f"Loaded {len(self._sessions)} weka session(s) from {self.trace_file} "
            f"(max_concurrent_sessions={self._max_concurrent_sessions}, block_size={self.block_size})"
        )

    def __str__(self):
        return self.name

    def _expand_prompt(self, chunked_hash: list[int], target_input_size: int) -> list[int]:
        """Expand compact block hashes into a unique, reproducible per-token id
        sequence of exactly target_input_size ids. Identical to
        Trace._expand_prompt (opal/workloads/workload.py) -- see there for the
        full rationale of the bit layout and caching.
        """
        raw_tokenized_prompt = []
        remaining = target_input_size
        for h in chunked_hash:
            expanded_size = self.block_size if remaining >= self.block_size else remaining
            key = f"{h}:{expanded_size}"

            if key not in self._expanded_generated_prompts:
                base = (hash(key) & self._hash_id_base_mask) << self._hash_id_offset_bits
                self._expanded_generated_prompts[key] = [(base + i) for i in range(expanded_size)]

            raw_tokenized_prompt.extend(self._expanded_generated_prompts[key])
            remaining -= expanded_size

            if remaining <= 0:
                break

        return raw_tokenized_prompt

    @staticmethod
    def _load_sessions(trace_file: str, limit: int = -1) -> list[dict]:
        if not os.path.isfile(trace_file) or os.path.getsize(trace_file) == 0:
            raise ValueError(f"Trace file {trace_file} does not exist or is empty")
        sessions = []
        with open(trace_file, "r") as f:
            for line in f:
                if limit >= 0 and len(sessions) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                session = json.loads(line)
                if session.get("requests"):
                    sessions.append(session)
        if not sessions:
            raise ValueError(f"No sessions with requests found in {trace_file}")
        return sessions

    def generate_requests(self):
        """Up to max_concurrent_sessions sessions are queued at once; a queued session only
        starts replaying once a currently-active session finishes and frees its slot (FIFO via
        the simpy.Resource semaphore -- see _replay_session)."""
        capacity = len(self._sessions) if self._max_concurrent_sessions < 0 else self._max_concurrent_sessions
        self.log.info(f"[WEKA REPLAY] [START] total sessions={len(self._sessions)} max_concurrency={capacity}")
        semaphore = simpy.Resource(self.simpy_env, capacity=capacity)
        active_processes = []

        for session in self._sessions:
            if self.opal_env.are_we_done():
                break
            proc = safe_process(self.simpy_env, self._replay_session(session, semaphore))
            active_processes.append(proc)

        for proc in active_processes:
            yield proc

        self.log.info(f"Weka replay finished with {self.request_id} requests")
        self.is_finished = True

    def _replay_session(self, session: dict, semaphore: simpy.Resource):
        trace_id = session.get("trace_id", "unknown")
        requests = session["requests"]

        with semaphore.request() as req:
            yield req
            session_sim_start = self.simpy_env.now
            self.log.debug(
                f"[weka_replay][session={trace_id}] acquired semaphore at sim_time={session_sim_start:.6f}s"
                f" | turns={len(requests)}"
            )

            for turn_idx, turn in enumerate(requests):
                if self.opal_env.are_we_done():
                    break

                if turn_idx == 0:
                    # Sleep until the first turn's recorded offset from session start.
                    sleep_time = float(turn["t"]) * self.multiplier_to_sec - (
                        self.simpy_env.now - session_sim_start
                    )
                else:
                    # Sleep for the recorded gap since the previous turn's completion.
                    sleep_time = float(turn.get("think_time", 0.0)) * self.multiplier_to_sec * self.inter_turn_multiplier

                if sleep_time > 0:
                    yield self.simpy_env.timeout(sleep_time)

                input_length = turn["input_length"]
                output_length = turn["output_length"]
                hash_ids = self._expand_prompt(turn["hash_ids"], input_length)
                assert len(hash_ids) == input_length, (
                    f"session {trace_id} turn {turn_idx}: "
                    f"expanded {len(hash_ids)} token ids but input_length={input_length}"
                )

                sim_submit_time = self.simpy_env.now
                request = LLMRequest(
                    self.simpy_env, self.stage_id, input_length, hash_ids=hash_ids, output_length=output_length
                )
                request.session_id = trace_id
                request.trace_id = trace_id
                request.span_id = turn_idx

                self.request_id += 1
                self.generated_requests += 1
                self.log.debug(
                    f"[WEKA REPLAY] [Session {trace_id}, turn {turn_idx + 1}/{len(requests)}] submitted"
                    f" | req_id={request.id} in={input_length} out={output_length}"
                    f" sleep_time={sleep_time:.6f}s sim_submit_time={sim_submit_time:.6f}s"
                )

                yield self.req_router.input_queue.put(request)

                # Wait for turn i to complete before sleeping think_time and submitting turn i+1.
                yield request.has_completed

            self.log.info(f"[WEKA REPLAY] [Session {trace_id}] finished")
