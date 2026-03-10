"""Benchmark tests for tinyros message passing speed with CPU/GPU payloads.

Run with: pytest -m run_explicitly tests/benchmark/tinyros/test_speed_tinyros.py
"""

import csv
import os
import socket
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import portal
import pytest

from tests.benchmark.tinyros.test_network import (Nodes, Topics,
                                                  build_network_config)
from tinyros import TinyNode

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
IMG_DIR = os.path.join(RESULTS_DIR, "images")
CSV_DIR = os.path.join(RESULTS_DIR, "csv")

REPETITIONS = 10000
VISUALIZE = True
WARMUP = 10
SLEEP_BETWEEN_ITERS_S = 1e-3

# Keep GPU assignment explicit and configurable at module level.
PUBLISHER_GPU_ID = "0"
SUBSCRIBER_GPU_ID = "1"


def save_latency_plot(
    *,
    lat_ms: Sequence[float],
    out_path: str | os.PathLike,
    title: str,
    shape: tuple[int, int],
    pub_hw: str,
    sub_hw: str,
    nbytes: int,
    dpi: int = 150,
    show_p50_p95: bool = True,
) -> None:
    """Save a PNG plot of latency (ms) over iterations."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y = np.asarray(lat_ms, dtype=np.float64)
    x = np.arange(len(y), dtype=np.int32)

    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel("Iteration")
    plt.ylabel("Latency [ms]")

    header = (
        f"{title}\n"
        f"{pub_hw} -> {sub_hw} | shape={shape[0]}x{shape[1]} | bytes={nbytes}"
    )
    plt.title(header)

    if show_p50_p95 and len(y) > 0:
        median = float(np.percentile(y, 50))
        mean = float(np.mean(y))
        p95 = float(np.percentile(y, 95))
        plt.axhline(median, linestyle="--", linewidth=1, color="red")
        plt.axhline(mean, linestyle="--", linewidth=1, color="green")
        plt.axhline(p95, linestyle="--", linewidth=1)
        plt.legend(
            [
                "latency",
                f"median={median:.3f}ms",
                f"mean={mean:.3f}ms",
                f"p95={p95:.3f}ms",
            ],
            loc="best",
        )

    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def get_free_port() -> int:
    """Find a free port on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port: int = s.getsockname()[1]
    s.close()
    return port


def wait_port_free(port: int, *, timeout_s: float = 2.0) -> None:
    """Wait until the given port is free (closed)."""
    t0 = time.perf_counter()
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.1)
            ok = s.connect_ex(("127.0.0.1", port))
        if ok != 0:
            return
        if time.perf_counter() - t0 > timeout_s:
            raise AssertionError(f"Port {port} still open after {timeout_s}s")
        time.sleep(0.01)


class SinkNode(TinyNode):
    """Subscriber node with optional GPU staging."""

    def __init__(self, *, sub_hw: str, pub_port: int, sub_port: int) -> None:
        self.sub_hw = sub_hw
        self._gpu_device: Any = None
        self._jax: Any = None

        if self.sub_hw == "gpu":
            import jax

            devices = jax.devices("gpu")
            if not devices:
                raise RuntimeError(
                    "No GPU devices visible to JAX in subscriber")
            self._jax = jax
            self._gpu_device = devices[0]

        super().__init__(
            name=Nodes.SUBSCRIBER,
            network_config=build_network_config(
                pub_port=pub_port,
                sub_port=sub_port),
        )

    def on_msg(self, msg: np.ndarray) -> float:
        """Return receive timestamp after optional host->device transfer."""
        if self.sub_hw == "gpu":
            arr = self._jax.device_put(msg, self._gpu_device)
            self._jax.block_until_ready(arr)
        return time.monotonic()


class PublisherNode(TinyNode):
    """Publisher node for the benchmark."""

    def __init__(self, *, pub_port: int, sub_port: int) -> None:
        self.ready = False
        super().__init__(
            name=Nodes.PUBLISHER,
            network_config=build_network_config(
                pub_port=pub_port,
                sub_port=sub_port),
        )

    def on_ready(self, _: float) -> None:
        """Mark publisher as ready once subscriber heartbeat is received."""
        self.ready = True


def _subscriber_worker(
    sub_hw: str,
    pub_port: int,
    sub_port: int,
) -> None:
    """Run subscriber process until stop signal is received."""
    sub: SinkNode | None = None
    try:
        if sub_hw == "gpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = SUBSCRIBER_GPU_ID

        sub = SinkNode(sub_hw=sub_hw, pub_port=pub_port, sub_port=sub_port)
        # One-shot readiness notification to publisher.
        sub.publish(Topics.READY, 1.0)
        while True:
            time.sleep(1e6)
    finally:
        if sub is not None:
            sub.shutdown()


@pytest.mark.run_explicitly
@pytest.mark.parametrize("pub_hw", ["cpu", "gpu"])
@pytest.mark.parametrize("sub_hw", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (2, 2),
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ],
)
def test_latency_cpu_gpu_payloads(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int],
    pub_hw: str,
    sub_hw: str,
) -> None:
    """Measure end-to-end latency between multiprocess publisher/subscriber."""
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    sub_port = get_free_port()
    pub_port = get_free_port()

    sub_proc: Any = None
    pub: PublisherNode | None = None

    try:
        sub_proc = portal.Process(
            _subscriber_worker,
            sub_hw,
            pub_port,
            sub_port,
            name="tinyros_subscriber_worker",
            start=True,
        )

        if pub_hw == "gpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = PUBLISHER_GPU_ID

        arr = np.zeros(shape, dtype=np.float32)
        payload: np.ndarray | Any = arr
        jax_mod: Any = None

        if pub_hw == "gpu":
            import jax

            devices = jax.devices("gpu")
            if not devices:
                pytest.skip("No GPU devices visible to JAX in publisher")

            jax_mod = jax
            payload = jax.device_put(arr, devices[0])
            jax.block_until_ready(payload)

        # Disable automatic atexit registration in the parent process only.
        # The subscriber runs in a separate process and is intentionally left
        # unpatched.
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        pub = PublisherNode(pub_port=pub_port, sub_port=sub_port)

        t_ready = time.monotonic()
        while not pub.ready:
            if time.monotonic() - t_ready > 20.0:
                raise AssertionError("Subscriber readiness topic timed out")
            time.sleep(0.001)

        latencies: list[float] = []

        for i in range(WARMUP + REPETITIONS):
            if pub_hw == "gpu":
                payload_host = np.asarray(payload)
                if jax_mod is not None:
                    jax_mod.block_until_ready(payload)
            else:
                payload_host = payload

            t0 = time.monotonic()
            futures = pub.publish(Topics.PAYLOAD, payload_host)
            if not futures:
                raise AssertionError(
                    "Publisher has no subscriber futures for topic")

            recv_ts = float(futures[0].result())
            if i >= WARMUP:
                latencies.append(recv_ts - t0)

            time.sleep(SLEEP_BETWEEN_ITERS_S)

        nbytes = int(arr.nbytes)
    finally:
        if pub is not None:
            try:
                pub.shutdown()
            except Exception:
                pass

        if sub_proc is not None:
            try:
                sub_proc.kill(timeout=5)
                sub_proc.join(timeout=5)
            except Exception:
                pass

        wait_port_free(pub_port)
        wait_port_free(sub_port)

    lat_ms = [x * 1e3 for x in latencies]

    # Save per-iteration plot
    if VISUALIZE:
        img_path = os.path.join(
            IMG_DIR,
            f"tinyros_latency_trace_{pub_hw}_to_{sub_hw}_{shape[0]}x{shape[1]}.png",
        )
        save_latency_plot(
            lat_ms=lat_ms,
            out_path=img_path,
            title="TinyROS latency trace",
            shape=shape,
            pub_hw=pub_hw,
            sub_hw=sub_hw,
            nbytes=nbytes,
        )

    stats = {
        "min": min(lat_ms),
        "max": max(lat_ms),
        "mean": statistics.mean(lat_ms),
        "std": statistics.stdev(lat_ms) if len(lat_ms) > 1 else 0.0,
        "median": statistics.median(lat_ms),
        "p95_best": statistics.quantiles(lat_ms, n=20)[0],
        "p95_worst": statistics.quantiles(lat_ms, n=20)[18],
    }

    csv_path = os.path.join(
        CSV_DIR,
        f"tinyros_latency_{pub_hw}_to_{sub_hw}.csv",
    )
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                [
                    "pub_hw",
                    "sub_hw",
                    "height",
                    "width",
                    "bytes",
                    "min_ms",
                    "max_ms",
                    "mean_ms",
                    "std_ms",
                    "median_ms",
                    "p95_best_ms",
                    "p95_worst_ms",
                ]
            )

        w.writerow(
            [
                pub_hw,
                sub_hw,
                shape[0],
                shape[1],
                nbytes,
                stats["min"],
                stats["max"],
                stats["mean"],
                stats["std"],
                stats["median"],
                stats["p95_best"],
                stats["p95_worst"],
            ]
        )

    assert len(latencies) == REPETITIONS, "Did not receive all messages"
