#!/usr/bin/env bash
# Benchmark the TFLite models on an Android phone connected by USB (USB debugging on).
# Requires adb on the laptop. Usage:  bash run_phone_benchmark.sh tflite_models
set -e
MODELS_DIR=${1:-tflite_models}
REMOTE=/data/local/tmp
BIN=android_aarch64_benchmark_model
URL=https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model

# 1. Download the official benchmark tool (once)
if [ ! -f "$BIN" ]; then
  echo "Downloading $BIN ..."
  wget -q -O "$BIN" "$URL" || { echo "Download failed: get the Android benchmark binary from the LiteRT/TFLite 'benchmark tools' page"; exit 1; }
fi

# 2. Copy the tool and the models to the phone
adb push "$BIN" "$REMOTE/benchmark_model" > /dev/null
adb shell chmod +x "$REMOTE/benchmark_model"

OUT="results_phone.txt"
{
  echo "Phone: $(adb shell getprop ro.product.model | tr -d '\r')"
  echo "SoC:   $(adb shell getprop ro.soc.model | tr -d '\r')"
  echo "Android $(adb shell getprop ro.build.version.release | tr -d '\r')"
} | tee "$OUT"

# 3. Run each model: 1 thread, CPU with XNNPACK
for f in "$MODELS_DIR"/nn_N*_*.tflite; do
  name=$(basename "$f")
  adb push "$f" "$REMOTE/$name" > /dev/null
  echo "==================== $name ====================" | tee -a "$OUT"
  adb shell "$REMOTE/benchmark_model" \
      --graph="$REMOTE/$name" \
      --num_threads=1 \
      --use_xnnpack=true \
      --warmup_runs=500 \
      --num_runs=10000 2>&1 | grep -E "Inference timings|Memory footprint|overall" | tee -a "$OUT"
done
echo "Saved $OUT"
