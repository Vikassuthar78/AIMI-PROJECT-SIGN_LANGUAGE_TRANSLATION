import cv2
import numpy as np
import tensorflow.lite as tflite

# -----------------------------
# Load TFLite Model
# -----------------------------
interpreter = tflite.Interpreter(model_path="../models/sign_model_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]['index']
output_index = output_details[0]['index']

labels = ["BYE", "HELLO", "HELP", "NAME", "THANK YOU", "VIKAS"]

# Required input size of your model
IMG_SIZE = 192

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # -----------------------------
    # 1. Preprocess Frame
    # -----------------------------
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    input_data = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    # -----------------------------
    # 2. Run Inference
    # -----------------------------
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)[0]
    idx = np.argmax(preds)
    sign = labels[idx]

    # -----------------------------
    # 3. Display Result
    # -----------------------------
    cv2.putText(frame, sign, (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 0), 3)

    cv2.imshow("SIGN LANGUAGE DETECTION", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
