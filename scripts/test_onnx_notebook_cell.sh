%%bash
set -euo pipefail

# Jupyter shell cell to test exported ONNX model.
# Usage: paste this cell into a notebook code cell and run.
MODEL="${MODEL:-runs-cls/yolov8n-cls-fashion/weights/best.onnx}"

if [ ! -f "$MODEL" ]; then
  echo "ERROR: ONNX model not found at: $MODEL" >&2
  echo "Check the run name / weights folder or set MODEL env var to the correct path."
  exit 1
fi

python3 - <<'PY'
import os, sys, numpy as np
try:
    import onnxruntime as ort
except Exception as e:
    print("ERROR: onnxruntime is not installed:", e, file=sys.stderr)
    sys.exit(2)

model_path = os.environ.get("MODEL", "runs-cls/yolov8n-cls-fashion/weights/best.onnx")
print("Loading ONNX model:", model_path)
sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# Examine inputs
inputs = sess.get_inputs()
if not inputs:
    print("Model has no inputs. Exiting.", file=sys.stderr); sys.exit(3)
print("Model inputs:")
for i in inputs:
    print(" -", i.name, "shape:", i.shape, "type:", i.type)

# Build a safe dummy tensor for the first input
inp = inputs[0]
shape = []
for dim in inp.shape:
    if isinstance(dim, str) or dim is None or (isinstance(dim, int) and dim <= 0):
        # common default for image models: [1,3,224,224]
        # if dynamic, set batch=1 and channels=3, H/W fallback 224
        shape = [1, 3, 224, 224]
        break
    else:
        shape.append(int(dim))
if not shape:
    shape = [1, 3, 224, 224]

# If first dim looks like (N, C, H, W) but has unexpected ordering, still attempt N,C,H,W
if len(shape) == 4:
    dtype = np.float32
    dummy = np.random.rand(*shape).astype(dtype)
else:
    # fallback to a flat vector
    dummy = np.random.rand(*shape).astype(np.float32)

print("Running inference with dummy input shape:", dummy.shape)
try:
    outputs = sess.run(None, {inp.name: dummy})
except Exception as e:
    print("ERROR during inference:", e, file=sys.stderr)
    sys.exit(4)

print("Inference successful. Number of outputs:", len(outputs))
for idx, out in enumerate(outputs):
    print(f" - output[{idx}] shape: {np.shape(out)} dtype: {getattr(out, 'dtype', type(out))}")

print("ONNX model smoke test passed.")
PY