# =============================
# models/convert_to_tflite.py
# =============================
import os, tensorflow as tf
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "models")
SRC = os.path.join(MODEL_DIR, "sign_model_best.keras")

if not os.path.exists(SRC):
    raise FileNotFoundError("sign_model_best.keras not found. Train first.")

best = tf.keras.models.load_model(SRC)

# float32
f32_path = os.path.join(MODEL_DIR, "sign_model_float32.tflite")
conv = tf.lite.TFLiteConverter.from_keras_model(best)
with open(f32_path, "wb") as f: f.write(conv.convert())

# int8
def rep():
    import numpy as np
    for _ in range(100):
        yield [np.random.uniform(-1,1,(1,192,192,3)).astype("float32")]
i8_path = os.path.join(MODEL_DIR, "sign_model_int8.tflite")
conv = tf.lite.TFLiteConverter.from_keras_model(best)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep
conv.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS
]
with open(i8_path, "wb") as f: f.write(conv.convert())

print("✅ Wrote:", f32_path, "and", i8_path)
