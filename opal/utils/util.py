# SPDX-License-Identifier: Apache-2.0
import re
from enum import IntEnum
import numpy as np


class KVC_TIER_TYPE(IntEnum):
    GPU = 1
    CPU = 2
    NVME = 3
    DFS = 4
    DISABLE = 5


def parse_bandwidth(s: str):
    """
    Parses a string containing a bandwidth/throughput value and unit.

    Args:
        s (str): Input string (e.g., '100 Mbps', '1.5 Gbit/s', '200 KB/s').

    Returns:
        dict: { 'value': float, 'unit': str }
              Unit normalized to a standard short form.
    """
    # Normalize input
    s = s.strip().replace("/s", "ps")  # unify /s to ps (per second)

    # Regex to capture numeric value + unit
    pattern = re.compile(r"([0-9]*\.?[0-9]?)\s*([A-Za-z]+)")
    match = pattern.search(s)

    if not match:
        try:
            value = float(s)
            unit = None
        except:
            raise ValueError(f"Cannot parse bandwidth string: {s}")
    else:
        value = float(match.group(1))
        unit = match.group(2).lower()
        # Normalize units
        unit = unit.replace("sec", "s")
        unit_map = {
            "bps": "bps",
            "bitps": "bps",
            "Bps": "Bps",
            "kbitps": "kbps",
            "kbps": "kbps",
            "KBps": "KBps",
            "mbps": "mbps",
            "mbitps": "mbps",
            "MBps": "MBps",
            "gbps": "gbps",
            "gbitps": "gbps",
            "GBps": "GBps",
            "tbps": "tbps",
            "tbitps": "tbps",
            "TBps": "TBps",
        }
        unit = unit_map.get(unit, unit)  # fallback to raw if unknown

    return (value, unit)


# This function is meant to catch and print the exceptions from the co-routine executions
import traceback
import simpy


def get_with_timeout(env: simpy.Environment, store: simpy.Store, timeout: float):
    """
    Attempts to get an object from a SimPy Store.
    Returns the object if successful within 'timeout' duration,
    otherwise returns None.
    """
    # Use Store.get() as a context manager to ensure
    # the request is canceled if the timeout occurs first.
    with store.get() as get_req:
        # Create a timeout event
        timeout_event = env.timeout(timeout)

        # Wait until either the get request triggers or the timeout triggers
        result = yield get_req | timeout_event

        if get_req in result:
            # Success: return the value from the store request
            return get_req.value
        else:
            # Timeout: return None
            return None


def safe_process(env, coro):
    """
    Run a SimPy coroutine asynchronously and print any exceptions.

    This wrapper provides error handling for SimPy generator functions (coroutines).

    Usage Patterns:
    ---------------
    1. Fire-and-forget (asynchronous):
       env.process(safe_process(env, my_coroutine()))
       # Spawns independent process, doesn't wait

    2. Wait for completion (synchronous):
       yield safe_process(env, my_coroutine())
       # Waits for process to complete, gets return value

    3. Inline execution (most common in OPAL):
       yield from my_coroutine()
       # Executes inline, no separate process created

    Understanding `yield` vs `yield from`:
    --------------------------------------

    `yield` - Pause and return ONE value:
        def task(env):
            yield env.timeout(5)  # Wait for 5 time units
            return "done"

    `yield from` - Delegate to ENTIRE generator:
        def sub_task(env):
            yield env.timeout(2)
            yield env.timeout(3)
            return "result"

        def main_task(env):
            result = yield from sub_task(env)  # Execute all of sub_task inline
            # Equivalent to writing sub_task's code directly here

    Key Difference:
    - `yield` waits for ONE event
    - `yield from` executes ENTIRE sub-generator inline

    Why `yield from` is used in vllm_worker.py:
    -------------------------------------------
    In _vllm_scheduling_loop(), we use:
        yield from self._scheduler_step()

    Instead of:
        yield safe_process(env, self._scheduler_step())

    Reasons:
    1. Interrupt Control: With `yield from`, interrupts propagate through the
       inline code, allowing us to catch them with try-except

    2. Simpler Code: No need to create separate process objects

    3. Lower Overhead: Single process instead of parent + child

    4. Atomic Execution: We can ensure _scheduler_step() completes without
       interruption by catching interrupts in the parent

    Example - Interrupt Handling:
    -----------------------------
    With `yield from` (current):
        try:
            yield from self._scheduler_step()  # Inline execution
        except simpy.Interrupt:
            pass  # Can catch interrupts from inside _scheduler_step()

    With `yield safe_process()` (alternative):
        try:
            yield safe_process(env, self._scheduler_step())  # Separate process
        except simpy.Interrupt:
            pass  # Only catches interrupts to parent, NOT to child process!

    The child process would be isolated and continue running even if parent
    is interrupted, potentially causing race conditions.

    SimPy Process Mechanics:
    ------------------------
    * env.process(generator()) returns a Process object
    * yield process returns the value returned by the generator
    * That return value is stored internally in process.value
    * But you only use .value if you did not yield it

    Exception Handling:
    -------------------
    * simpy.Interrupt: Re-raised (normal control flow, not an error)
    * Other exceptions: Caught, printed, and simulation stopped
    """

    def _runner():
        try:
            ret = yield from coro
            return ret
        except simpy.Interrupt:
            # Interrupt is a normal control flow mechanism in SimPy, re-raise it
            # This allows interrupt-driven wake-up patterns (e.g., sleeping processes)
            raise
        except Exception:
            msg = f"💥 Exception in process {coro}:"
            print(msg)
            traceback.print_exc()
            raise simpy.core.StopSimulation()

    return env.process(_runner())


def get_bool_env_var(name: str, default: bool = False) -> bool:
    import os

    value = os.getenv(name)

    if value is None:
        return default

    value = value.strip().lower()

    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise ValueError(f"Invalid boolean value for {name}: {value}")


# safely parsing bool from json string
def parse_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return False


def generate_time_with_rate_variation(request_rate_per_sec=1.0, jitter=1.0):
    # Base exponential sample
    mean = 1.0 / request_rate_per_sec
    interval = np.random.exponential(scale=mean)
    # Apply additional variation if needed with jitter
    # jitter 1.0 = pure exponential
    # jitter 0.0 = deterministic fixed
    return mean * (1 - jitter) + interval * jitter


def sample_series_K(series, k: int = 1):
    assert len(series) > 0
    arr = np.array(series)
    merged = np.array([arr[i : i + k].mean() for i in range(0, len(arr), k)])
    return merged


def normalize_str(key: str) -> str:
    """Normalize text: lowercase + collapse multiple spaces."""
    return " ".join(key.lower().split())


def check_and_create_directory(dir_path: str, create_parents=True, fail_if_exists=True):
    from pathlib import Path

    path = Path(dir_path)
    # if it exists, then just dont do anything (exist_ok = True)
    path.mkdir(parents=create_parents, exist_ok=fail_if_exists)


import simpy


class BlockingStore:
    def __init__(self, env, capacity):
        self.env = env
        self.capacity = capacity
        self.items = []

    def put(self, item):
        while len(self.items) >= self.capacity:
            # block until space is available
            yield self.env.timeout(1)
        self.items.append(item)

    def get(self):
        while len(self.items) == 0:
            # block until an item is available
            yield self.env.timeout(1)
        return self.items.pop(0)


def format_decimal_nonzero(x: float, positions: int = 2) -> str:
    s = f"{x:.20f}"  # enough precision
    integer, decimal = s.split(".")
    nonzero = ""
    for ch in decimal:
        if ch != "0":
            nonzero += ch
            if len(nonzero) == positions:
                break
    return f"{integer}.{'0' * (decimal.find(nonzero[0]))}{nonzero}"


if __name__ == "__main__":
    # Example usage:
    print(parse_bandwidth("100 Mbps"))  # {'value': 100.0, 'unit': 'mbps'}
    print(parse_bandwidth("1.5 Gbit/s"))  # {'value': 1.5, 'unit': 'gbps'}
    print(parse_bandwidth("2.5 MB/s"))  # {'value': 1.5, 'unit': 'MBps'}
    print(parse_bandwidth("3.5 MB/sec"))  # {'value': 1.5, 'unit': 'MBps'}
    print(parse_bandwidth("200. KB/s"))  # {'value': 200.0, 'unit': 'KBps'}
    print(parse_bandwidth("1123.312"))  # {'value': 100.0, 'unit': ''}
