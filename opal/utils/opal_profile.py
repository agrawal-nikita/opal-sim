# SPDX-License-Identifier: Apache-2.0
import cProfile
import pstats
import io


def profile_function(func):
    """Profile a given function and print sorted stats."""
    profiler = cProfile.Profile()
    profiler.enable()  # start profiling
    ret = func()  # run the function
    profiler.disable()  # stop profiling

    # Create an in-memory text stream for the stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)  # show top 10 time-consuming calls
    print(s.getvalue())
    return ret


def test_profiler():
    def slow_function():
        """Example function that does some slow operations."""
        total = 0
        for i in range(1, 10000):
            for j in range(1, 1000):
                total += (i * j) % 7
        return total

    profile_function(slow_function)


# Run the profiler
if __name__ == "__main__":
    test_profiler()
