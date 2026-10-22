# ============================================================
# convert_to_onnx.py
# Converts all-MiniLM-L6-v2 SentenceTransformer to ONNX INT8.
# ============================================================

from sentence_transformers import SentenceTransformer
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
import os

MODEL_NAME = "all-MiniLM-L6-v2"
ONNX_PATH = "/home/pesu-rf/Documents/Capstone_Team165/partition/subg/model.onnx"
INT8_PATH = "/home/pesu-rf/Documents/Capstone_Team165/partition/subg/model_int8.onnx"

print(f"Loading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME, device="cpu")

# Export to ONNX
print("Exporting to ONNX...")
from sentence_transformers.util import export_onnx

export_onnx(
    model,
    ONNX_PATH,
    opset=17,
    device="cpu",
)
print(f"Saved ONNX model to {ONNX_PATH}")

# Quantize to INT8
print("Quantizing to INT8...")
quantize_dynamic(
    model_input=ONNX_PATH,
    model_output=INT8_PATH,
    weight_type=QuantType.QInt8,
    optimize_model=True,
)
print(f"Saved INT8 model to {INT8_PATH}")

# Verify
import onnxruntime as ort
import numpy as np

print("Verifying INT8 model...")
sess = ort.InferenceSession(INT8_PATH, providers=["CPUExecutionProvider"])
dummy = np.array([["hello world", "test sentence"]])
outputs = sess.run(None, {"sentence": dummy})
print(f"Output shape: {outputs[0].shape}")
print("DONE.")
