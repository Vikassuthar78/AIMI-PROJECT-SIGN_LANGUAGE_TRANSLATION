from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import threading
import queue
import time
import tensorflow as tf
import os

app = Flask(__name__)

# ----------------------------
# ✅ Absolute paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "sign_model_int8.tflite")
dataset_path = os.path.join(BASE_DIR, "..", "two_hand_dataset")

print("✅ Model path:", model_path)
print("✅ Dataset path:", dataset_path)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

# ----------------------------
# ✅ Load TFLite Model
# ----------------------------
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]["shape"]
img_size = input_shape[1]   # e.g. 64

print(f"✅ TFLite Model Loaded: expects {img_size}x{img_size} image")

# Load class names from dataset folder
gesture_classes = sorted([
    d for d in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, d))
])
num_classes = len(gesture_classes)

print("✅ Gestures:", gesture_classes)

# ----------------------------
# ✅ Mediapipe
# ----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.4)
mp_draw = mp.solutions.drawing_utils

# ----------------------------
# ✅ Globals
# ----------------------------
cap = cv2.VideoCapture(0)
frame_queue = queue.Queue(maxsize=3)
current_gesture = "Waiting..."

# ----------------------------
# ✅ Capture Thread
# ----------------------------
def capture_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        if not frame_queue.full():
            frame_queue.put(frame)
        time.sleep(0.005)

threading.Thread(target=capture_frames, daemon=True).start()

# ----------------------------
# ✅ Prediction Thread
# ----------------------------
def predict_loop():
    global current_gesture

    while True:
        try:
            frame = frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            current_gesture = "No Hand"
            continue

        h, w, _ = frame.shape
        preds_total = np.zeros(num_classes)

        for hand_landmarks in results.multi_hand_landmarks:
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]

            x_min = int(min(xs) * w)
            x_max = int(max(xs) * w)
            y_min = int(min(ys) * h)
            y_max = int(max(ys) * h)

            # ✅ Square bounding box
            size = int(max(x_max - x_min, y_max - y_min) * 1.3)
            cx = int((x_min + x_max) / 2)
            cy = int((y_min + y_max) / 2)

            x1 = max(0, cx - size // 2)
            y1 = max(0, cy - size // 2)
            x2 = min(w, cx + size // 2)
            y2 = min(h, cy + size // 2)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (img_size, img_size))
            crop = crop.astype(np.float32) / 255.0
            crop = np.expand_dims(crop, axis=0)

            interpreter.set_tensor(input_details[0]["index"], crop)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]["index"])[0]

            preds_total += preds

        preds_total /= len(results.multi_hand_landmarks)
        idx = np.argmax(preds_total)
        confidence = preds_total[idx] * 100

        current_gesture = f"{gesture_classes[idx]} ({confidence:.1f}%)"

threading.Thread(target=predict_loop, daemon=True).start()

# ----------------------------
# ✅ Frame Stream
# ----------------------------
def generate_frames():
    global current_gesture

    while True:
        try:
            frame = frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        cv2.putText(frame, current_gesture, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# ----------------------------
# ✅ Flask Routes
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/get_prediction")
def get_prediction():
    return jsonify({"gesture": current_gesture})

# ----------------------------
# ✅ Run App
# ----------------------------
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
