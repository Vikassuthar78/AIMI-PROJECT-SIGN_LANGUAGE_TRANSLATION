# =====================
# src/predictor.py
# =====================
import os, json, numpy as np, cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "models")
LABELS = os.path.join(MODEL_DIR, "label_map.json")

class Predictor:
    def __init__(self):
        with open(LABELS, "r", encoding="utf-8") as f:
            self.idx2label = {str(int(k)): v for k,v in json.load(f).items()}
        self.num_classes = len(self.idx2label)
        self.input_size = (192,192)
        self.preproc = "mnv2"
        self.backend = None

        self._load_backend()

    def _load_backend(self):
        self.interp = None
        self.model = None
        try:
            import tensorflow as tf
            int8 = os.path.join(MODEL_DIR, "sign_model_int8.tflite")
            f32  = os.path.join(MODEL_DIR, "sign_model_float32.tflite")
            if os.path.exists(int8):
                self.interp = tf.lite.Interpreter(model_path=int8)
                self.interp.allocate_tensors()
                self.input_index  = self.interp.get_input_details()[0]["index"]
                self.output_index = self.interp.get_output_details()[0]["index"]
                shp = self.interp.get_input_details()[0]["shape"]
                self.input_size = (int(shp[1]), int(shp[2]))
                self.backend = "tflite_int8"; return
            if os.path.exists(f32):
                self.interp = tf.lite.Interpreter(model_path=f32)
                self.interp.allocate_tensors()
                self.input_index  = self.interp.get_input_details()[0]["index"]
                self.output_index = self.interp.get_output_details()[0]["index"]
                shp = self.interp.get_input_details()[0]["shape"]
                self.input_size = (int(shp[1]), int(shp[2]))
                self.backend = "tflite_float32"; return
            # fallback to .keras
            from tensorflow.keras.models import load_model
            keras_path = os.path.join(MODEL_DIR, "sign_model_best.keras")
            self.model = load_model(keras_path)
            self.input_size = tuple(self.model.input_shape[1:3])
            self.backend = "keras"
        except Exception as e:
            raise RuntimeError(f"Failed to load model backend: {e}")

    def preprocess(self, img_bgr):
        h,w = self.input_size
        x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (w,h), interpolation=cv2.INTER_AREA).astype("float32")
        x = (x/127.5)-1.0  # MobileNetV2
        return np.expand_dims(x,0)

    def predict(self, img_bgr):
        x = self.preprocess(img_bgr)
        if self.backend.startswith("tflite"):
            self.interp.set_tensor(self.input_index, x.astype(np.float32))
            self.interp.invoke()
            out = self.interp.get_tensor(self.output_index)[0]
        else:
            out = self.model.predict(x, verbose=0)[0]
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        s = out.sum()
        return out/s if s>0 else np.ones(self.num_classes)/self.num_classes
