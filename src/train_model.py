# =========================
# src/train_model.py
# =========================
# MobileNetV2 training for ISL (1- and 2-hand signs)
# Outputs:
#   ../models/sign_model_best.keras
#   ../models/sign_model_final.keras
#   ../models/sign_model_float32.tflite
#   ../models/sign_model_int8.tflite
#   ../models/label_map.json

# ---- set env BEFORE importing tensorflow (stability on Windows/CPU) ----
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["KMP_BLOCKTIME"] = "0"

import sys, json, random
import numpy as np
from collections import Counter
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# ----------------------------
# Paths
# ----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(ROOT, "two_hand_dataset")
MODEL_DIR   = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.isdir(DATASET_DIR) or len([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR,d))]) == 0:
    print(f"❌ Dataset missing or empty: {DATASET_DIR}")
    sys.exit(1)

# ----------------------------
# Hyperparameters
# ----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)

IMG_SIZE   = 192
BATCH_SIZE = 16
VAL_SPLIT  = 0.2
EPOCHS_FROZEN = 8
EPOCHS_FINE   = 10
AUG = 0.25

# ----------------------------
# Data pipeline
# ----------------------------
train_val_gen = ImageDataGenerator(
    preprocessing_function=mobilenet_v2.preprocess_input,
    validation_split=VAL_SPLIT,
    rotation_range=15,
    width_shift_range=AUG,
    height_shift_range=AUG,
    zoom_range=0.25,
    shear_range=0.1,
    brightness_range=(0.80, 1.20),
    horizontal_flip=True
)

train_gen = train_val_gen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    subset="training",
    seed=SEED
)

val_gen = train_val_gen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    subset="validation",
    seed=SEED
)

num_classes = train_gen.num_classes
class_indices = train_gen.class_indices
index_to_label = {v: k for k, v in class_indices.items()}

with open(os.path.join(MODEL_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(index_to_label, f, ensure_ascii=False, indent=2)
print("✅ label_map.json saved:", index_to_label)

counts = Counter(train_gen.classes)
max_count = max(counts.values())
class_weight = {cls: float(max_count / count) for cls, count in counts.items()}
print("Class counts:", counts)
print("Class weights:", class_weight)

# ----------------------------
# Build model (MobileNetV2)
# ----------------------------
base = mobilenet_v2.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.35)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.35)(x)
out = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=base.input, outputs=out)

# Phase 1: freeze backbone
for l in base.layers:
    l.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

ckpt_path  = os.path.join(MODEL_DIR, "sign_model_best.keras")
final_path = os.path.join(MODEL_DIR, "sign_model_final.keras")

cbs = [
    ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    CSVLogger(os.path.join(MODEL_DIR, "training_log.csv"), append=False)
]

print("\n=== Phase 1: train head ===")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FROZEN,
    callbacks=cbs,
    class_weight=class_weight
)

# Phase 2: fine-tune top ~20%, keep BatchNorm frozen
def freeze_bn(m):
    for layer in m.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
freeze_bn(base)
for l in base.layers[int(len(base.layers)*0.8):]:
    if not isinstance(l, layers.BatchNormalization):
        l.trainable = True

try:
    print("\n=== Phase 2: fine-tune top backbone ===")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINE,
        callbacks=cbs,
        class_weight=class_weight
    )
except Exception as e:
    print("⚠️ Fine-tune skipped:", repr(e))

model.save(final_path)
print(f"✅ Saved: {ckpt_path} and {final_path}")

# ----------------------------
# Export to TFLite (float32 + int8)
# ----------------------------
def export_tflite(saved_model_path, out_float, out_int8):
    best = tf.keras.models.load_model(saved_model_path)

    # float32
    conv = tf.lite.TFLiteConverter.from_keras_model(best)
    tflite_f32 = conv.convert()
    with open(out_float, "wb") as f: f.write(tflite_f32)

    # int8 with representative dataset
    def rep_data():
        steps = min(100, len(val_gen))
        for _ in range(steps):
            xb, _ = next(val_gen)
            yield [xb.astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(best)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_data
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    tflite_int8 = conv.convert()
    with open(out_int8, "wb") as f: f.write(tflite_int8)

export_tflite(
    ckpt_path,
    os.path.join(MODEL_DIR, "sign_model_float32.tflite"),
    os.path.join(MODEL_DIR, "sign_model_int8.tflite")
)
print("📦 Exported TFLite models.")
