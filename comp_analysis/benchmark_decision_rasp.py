"""
Measure the time per handover decision of the NN-RL-RMSE model (TensorFlow Lite)
and of the SAW, WPM and TOPSIS methods (NumPy), under identical conditions.

Runs on the laptop and on the Raspberry Pi (only NumPy + a TFLite runtime needed):
    Laptop: python benchmark_decision.py --models tflite_models
    Pi:     pip install numpy ai-edge-litert
            python benchmark_decision.py --models tflite_models

All timings use a single CPU thread and include everything a real decision needs.
NN-fp32 / NN-int8: full decision as in the simulator (normalization of the 8 parameters,
velocity computation, inference and argmax). infer-fp32 / infer-int8: inference only.
"""
import argparse
import csv
import glob
import os
import platform
import re
import time

import numpy as np

# ---- TFLite runtime: LiteRT first, then the older packages ----
try:
    from ai_edge_litert.interpreter import Interpreter
    RUNTIME = "ai-edge-litert"
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        RUNTIME = "tflite-runtime"
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        RUNTIME = "tensorflow " + tf.__version__

M = 10                                            # number of decision attributes
WEIGHTS = np.full(M, 1 / M)
DIRECTIONS = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0], dtype=bool)  # 1 = benefit


# ============================ MCDM methods (one decision each) ============================
def saw(X):
    mn, mx = X.min(axis=0), X.max(axis=0)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    norm = np.where(DIRECTIONS, (X - mn) / rng, (mx - X) / rng)
    return int(np.argmax(norm @ WEIGHTS))


def wpm(X):
    mn, mx = X.min(axis=0), X.max(axis=0)
    norm = np.where(DIRECTIONS, X / mx, mn / X)
    return int(np.argmax(np.prod(norm ** WEIGHTS, axis=1)))


def topsis(X):
    norm = X / np.sqrt((X ** 2).sum(axis=0))
    V = norm * WEIGHTS
    ideal = np.where(DIRECTIONS, V.max(axis=0), V.min(axis=0))
    anti = np.where(DIRECTIONS, V.min(axis=0), V.max(axis=0))
    d_pos = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - anti) ** 2).sum(axis=1))
    denom = d_pos + d_neg
    return int(np.argmax(d_neg / np.where(denom == 0, 1.0, denom)))


# ============================ NN (TFLite) ============================
class NNDecision:
    def __init__(self, path):
        self.interp = Interpreter(model_path=path, num_threads=1)
        self.interp.allocate_tensors()
        self.idx = {}
        for d in self.interp.get_input_details():
            if d["dtype"] == np.int32:
                self.idx["protocol"] = d["index"]
            elif "velocities" in d["name"]:
                self.idx["velocities"] = d["index"]
            else:
                self.idx["inputs"] = d["index"]
        self.out = self.interp.get_output_details()[0]["index"]

    def infer(self, protocol, velocities, inputs):
        """Inference only: copy the inputs, run the network, select the best network."""
        self.interp.set_tensor(self.idx["protocol"], protocol)
        self.interp.set_tensor(self.idx["velocities"], velocities)
        self.interp.set_tensor(self.idx["inputs"], inputs)
        self.interp.invoke()
        return int(np.argmax(self.interp.get_tensor(self.out)))

    def __call__(self, sample):
        """Full decision, as in the simulator: preprocessing + inference."""
        raw, dist_hist, protocol = sample
        return self.infer(protocol, *preprocess(raw, dist_hist))


# Parameter order used by the model: RSSI, SNR, Throughput, BER, FEC, PC, MC, HC
NN_BENEFIT = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
VEL_FACTOR = 10.0 / 30.0   # distance per 100 ms -> m/s, divided by VEL_SCALE = 30 m/s


def preprocess(raw, dist_hist):
    """Same steps as normalizeInputs() and RMSE_RL() in the simulator."""
    # 1. Min-max normalization across candidates (0 = best value)
    mn, mx = raw.min(axis=0), raw.max(axis=0)
    rng = mx - mn
    const = rng == 0
    rng = np.where(const, 1.0, rng)
    norm = np.where(NN_BENEFIT, (mx - raw) / rng, (raw - mn) / rng)
    norm[:, const] = 0.0
    # 2. Relative velocities from the last 4 distances, scaled and clipped
    vel = np.clip(np.diff(dist_hist, axis=1) * VEL_FACTOR, -1.0, 1.0)
    return vel.astype(np.float32), norm.astype(np.float32)


class NNInferenceOnly:
    """Wrapper that times only the network, for the breakdown."""
    def __init__(self, nn):
        self.nn = nn

    def __call__(self, sample):
        return self.nn.infer(*sample)


# ============================ Timing ============================
def time_method(fn, samples, warmup, runs):
    for i in range(warmup):
        fn(samples[i % len(samples)])
    t = np.empty(runs)
    for i in range(runs):
        s = samples[i % len(samples)]
        start = time.perf_counter_ns()
        fn(s)
        t[i] = time.perf_counter_ns() - start
    t /= 1000.0  # microseconds
    return {"mean": t.mean(), "std": t.std(), "median": np.median(t),
            "p95": np.percentile(t, 95), "p99": np.percentile(t, 99)}


def make_samples(n, count, rng):
    # Raw (un-normalized) values, as received by the NN in the simulator
    nn_raw = []
    for _ in range(count):
        raw = rng.uniform(0.1, 1.0, (n, 8))                         # 8 observable parameters
        start = rng.uniform(10, 1000, (n, 1))                       # distance to each AP [m]
        dist_hist = start + np.cumsum(rng.uniform(-1.5, 1.5, (n, 4)), axis=1)
        protocol = rng.randint(0, 7, (n, 1)).astype(np.int32)
        nn_raw.append((raw, dist_hist, protocol))
    # Already preprocessed inputs, for the inference-only measurement
    nn_pre = [(p, *preprocess(raw, d)) for raw, d, p in nn_raw]
    mcdm = [rng.uniform(0.1, 1.0, (n, M)) for _ in range(count)]   # >0 for WPM
    return nn_raw, nn_pre, mcdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tflite_models")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--runs", type=int, default=10000)
    ap.add_argument("--out", default=None, help="CSV file (default: results_<host>.csv)")
    args = ap.parse_args()

    host = platform.node()
    print("=" * 96)
    print(f"Host: {host} | {platform.machine()} | {platform.processor() or platform.platform()}")
    print(f"Python {platform.python_version()} | NumPy {np.__version__} | Runtime: {RUNTIME}")
    print(f"Single thread | warm-up {args.warmup} | runs {args.runs}")
    print("=" * 96)

    models = {}
    for path in glob.glob(os.path.join(args.models, "nn_N*_*.tflite")):
        m = re.search(r"nn_N(\d+)_(\w+)\.tflite", os.path.basename(path))
        models.setdefault(int(m.group(1)), {})[m.group(2)] = path

    rng = np.random.RandomState(0)   # works with old NumPy versions too
    rows = []
    print(f"{'Method':<14}{'N':>4}{'mean':>10}{'std':>10}{'median':>10}{'p95':>10}{'p99':>10}   [us]")
    print("-" * 96)
    for n in sorted(models):
        nn_raw, nn_pre, mcdm_samples = make_samples(n, 256, rng)
        methods = [("SAW", saw, mcdm_samples), ("WPM", wpm, mcdm_samples),
                   ("TOPSIS", topsis, mcdm_samples)]
        for tag in ("fp32", "int8"):
            if tag in models[n]:
                nn = NNDecision(models[n][tag])
                methods.append((f"NN-{tag}", nn, nn_raw))                          # full decision
                methods.append((f"  infer-{tag}", NNInferenceOnly(nn), nn_pre))     # breakdown
        for name, fn, samples in methods:
            r = time_method(fn, samples, args.warmup, args.runs)
            rows.append({"host": host, "method": name, "N": n, **r})
            print(f"{name:<14}{n:>4}{r['mean']:>10.2f}{r['std']:>10.2f}{r['median']:>10.2f}"
                  f"{r['p95']:>10.2f}{r['p99']:>10.2f}")
        print("-" * 96)

    out = args.out or f"results_{host}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
