# SPDX-License-Identifier: Apache-2.0
import logging
import numpy as np
import matplotlib.pyplot as plt
from opal.utils.util import sample_series_K
from opal.stats.stage_statistics import StageStatistics
import os

logger = logging.getLogger("plot")


def safe_hist_bins(data, min_bins=1, max_bins=50):
    data = np.array(data)
    data_range = data.max() - data.min()

    if data_range < 1e-12:  # treat as constant
        return min_bins
    else:
        # Freedman-Diaconis
        n = len(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            bin_width = data_range / min(max_bins, n)
        else:
            bin_width = 2 * iqr / n ** (1 / 3)
        num_bins = int(np.ceil(data_range / bin_width))
        num_bins = max(min_bins, min(num_bins, max_bins))
        return num_bins


def save_plot(plt, fname):
    plt.tight_layout()
    plt.savefig(fname, format="pdf", dpi=300, bbox_inches="tight")


def plot_latency_histogram(stats: StageStatistics):
    # data = stats.get_histogram()
    fig = plt.figure(figsize=(8, 6))
    num_bins = safe_hist_bins(stats.raw_latency_values)
    counts, bin_edges, patches = plt.hist(
        stats.raw_latency_values, bins=num_bins, color="skyblue", edgecolor="blue", alpha=0.7, width=0.1, align="right"
    )
    # Add labels and title
    plt.gca().yaxis.get_major_locator().set_params(integer=True)
    plt.xlabel("Latency (sec)")
    plt.ylabel("Request count")
    plt.title("End to end LLM request latency distribution")
    # Display the plot
    plt.grid(True, alpha=0.3)
    return plt


def plot_cdf(stats: StageStatistics, name="CDF"):
    data = stats.raw_latency_values
    print(stats.raw_latency_values)
    # Sort data for CDF
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(data) + 1) / len(data)

    # Compute statistics
    mean_val = np.mean(data)
    median_val = np.median(data)
    p75 = np.percentile(data, 75)
    p90 = np.percentile(data, 90)
    p95 = np.percentile(data, 95)

    # Plot CDF
    plt.figure(figsize=(9, 5))
    plt.plot(sorted_data, cdf, marker="o", linestyle="-", color="blue", label="CDF")

    # Add vertical lines for statistics
    plt.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.3f}")
    plt.axvline(median_val, color="green", linestyle="--", label=f"Median: {median_val:.3f}")
    plt.axvline(p75, color="purple", linestyle="--", label=f"P75: {p75:.3f}")
    plt.axvline(p90, color="orange", linestyle="--", label=f"P90: {p90:.3f}")
    plt.axvline(p95, color="brown", linestyle="--", label=f"P95: {p95:.3f}")

    # Labels and title
    plt.xlabel("Request processing time in seconds")
    plt.ylabel("CDF")
    plt.title("CDF with Mean, Median, and Percentiles")
    plt.legend()
    plt.grid(alpha=0.3)
    return plt


def _plot_per_time_timeseries(series, per_K: int = 1, y_title="???/sec", plot_title="??? sampled per second"):
    req_per_sec = sample_series_K(series, per_K)
    # X-axis: elapsed time in seconds
    elapsed_time = [_ * per_K for _ in list(range(len(req_per_sec)))]

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(elapsed_time, req_per_sec, marker="", linestyle="-", color="blue")

    # Labels and title
    plt.xlabel(f"Elapsed Time (seconds), sampled_freq {per_K} secs")
    plt.ylabel(f"{y_title}")
    plt.title(f"{plot_title}")
    plt.gca().yaxis.get_major_locator().set_params(integer=True)

    # Optional grid
    plt.grid(True)
    return plt


def plot_system_throughput(stats: StageStatistics, per_K: int = 1):
    return _plot_per_time_timeseries(
        stats.per_unit_req_done, per_K, "LLM prefill requests/sec", f"Prefill requests done per {per_K} seconds"
    )


def plot_workers_per_sec(stats: StageStatistics, per_K: int = 1):
    return _plot_per_time_timeseries(
        stats.per_unit_workers, per_K, "total active workers", f"avg. active workers per {per_K} seconds"
    )


def plot_gpu_utilization_per_sec(stats: StageStatistics, per_K: int = 1):
    return _plot_per_time_timeseries(
        stats.per_unit_gpu_utilization, per_K, "Avg GPU Utilization %", f"avg. GPU utilization per {per_K} seconds"
    )


def plot_multiple_cdfs(stats: StageStatistics, title="CDF Comparison"):
    """
    Plot multiple CDFs on one figure.

    Parameters
    ----------
    series_list : list of array-like
        Each entry is a dataset (e.g., list or np.array of latencies).
    labels : list of str
        Labels for each dataset (same length as series_list).
    title : str
        Title for the plot.
    """

    series_list = [stats.raw_latency_values, stats.raw_queuing_values, stats.raw_ttft_values]
    labels = ["E2E", "queuing", "TTFT"]

    plt.figure(figsize=(9, 5))

    markers = ["o", "x", "", "^", "D", "*"]
    for i, (data, label) in enumerate(zip(series_list, labels)):
        data = np.asarray(data)
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(data) + 1) / len(data)

        plt.plot(sorted_data, cdf, linestyle="-", marker=markers[i % len(markers)], label=label)

    plt.xlabel("Time in seconds")
    plt.ylabel("CDF")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt


def plot_stacked_histogram(stats: StageStatistics, num_bins: int = 10):
    """Plot a stacked bar chart of the histogram with latency source breakdowns."""
    histogram_data, sources = stats.get_histogram(num_bins)

    # Prepare data for plotting
    bin_lows = [entry["bin_low"] for entry in histogram_data]
    bin_highs = [entry["bin_high"] for entry in histogram_data]
    bin_width = (bin_highs[0] - bin_lows[0]) * 0.8  # Adjust width for bars

    # Prepare stacked bar data
    bottom = np.zeros(num_bins)
    plt.figure(figsize=(10, 6))

    # Use a colormap to assign distinct colors to each source
    colors = plt.cm.get_cmap("tab20", len(sources))

    # Plot stacked bars
    for i, source in enumerate(sources):
        heights = [entry["average_breakdown"][source] * entry["count"] for entry in histogram_data]
        plt.bar(bin_lows, heights, bin_width, bottom=bottom, label=source, color=colors(i))
        bottom += np.array(heights)

    # Customize plot
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.title("Latency Histogram with Source Breakdown")
    plt.legend(title="Latency Sources")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def simend_plot(stats: StageStatistics, opalConfig: "OpalConfig", dir: str = "./"):
    # we plot the following
    # Request/s
    # CDF of TTFT, queuing time, and total E2E
    # per-second workers

    # 1st the CDF distribution
    plt = plot_multiple_cdfs(stats)
    file = "cdf-latencies.pdf"
    full_path = os.path.join(dir, file)
    save_plot(plt, os.path.join(dir, file))
    logger.info(f"Done plotting {full_path}")

    # 2nd system throughput
    plt = plot_system_throughput(stats, 1)
    file = "thrp-request-sec.pdf"
    full_path = os.path.join(dir, file)
    save_plot(plt, os.path.join(dir, file))
    logger.info(f"Done plotting {full_path}")

    plt = plot_workers_per_sec(stats, 1)
    file = "thrp-workers-sec.pdf"
    full_path = os.path.join(dir, file)
    save_plot(plt, os.path.join(dir, file))
    logger.info(f"Done plotting {full_path}")

    plt = plot_gpu_utilization_per_sec(stats, 1)
    file = "gpu-utilization-per-sec.pdf"
    full_path = os.path.join(dir, file)
    save_plot(plt, os.path.join(dir, file))
    logger.info(f"Done plotting {full_path}")

    # per-second workers
    plt = plot_latency_histogram(stats)
    file = "histo-latencies.pdf"
    full_path = os.path.join(dir, file)
    save_plot(plt, os.path.join(dir, file))
    logger.info(f"Done plotting {full_path}")
