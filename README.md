# ✋ AI Sign Language Detection System  
Real-Time Hand Gesture Recognition using MediaPipe, TensorFlow & TFLite

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green.svg)
![Flask](https://img.shields.io/badge/Flask-WebApp-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📌 Overview  
This project is a **real-time Sign Language Detection System** built using:

- ✅ MediaPipe Hands – 21 landmark detection  
- ✅ TensorFlow / Keras – for training gesture classifier  
- ✅ TFLite – for fast mobile/web deployment  
- ✅ Flask Web App – real-time webcam inference  

The system detects custom gestures like **HELLO, HELP, NAME, THANK YOU, BYE, VIKAS**, etc.

---

## ✨ Features

✅ Real-time hand gesture recognition  
✅ Custom dataset support  
✅ TFLite optimized model  
✅ Clean Flask UI with confidence meter  
✅ Easy-to-train pipeline  
✅ Multi-class gesture support  
✅ Works with any Laptop Camera  

---

## 📂 Project Structure

```
AIML PROJECT/
│── models/
│   ├── convert_to_tflite.py
│   ├── label_map.json
│   ├── sign_model_best.keras
│   ├── sign_model_final.keras
│   ├── sign_model_float32.tflite
│   ├── sign_model_int8.tflite
│   └── training_log.csv
│
│── two_hand_dataset/      (Dataset created from collect_data.py)
│── HELLO/
│── HELP/
│── NAME/
│── THANK YOU/
│── BYE/
│
│
│── webapp/
│   ├── app.py
│   ├── static/
│   └── templates/
│       └── index.html
│
│── src/
│── collect_data.py
│── train_model.py
│── test_hands.py
│── requirements.txt
│── run.bat
│── README.md
```

---

# ✅ Step-by-Step Guide (Complete Workflow)

This project follows a simple **3-stage AI pipeline**:

---

# 1️⃣ Create & Activate Virtual Environment

### ✅ Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### ✅ Linux / Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 2️⃣ Install Required Libraries
Install all dependencies:

```bash
pip install -r requirements.txt
```

---

# 3️⃣ Collect Data (Important Step)

Use **collect_data.py** to capture gesture data.

### ✅ Run data collection
```bash
cd src
python collect_data.py
```

### ✅ What happens:
- Webcam opens  
- MediaPipe extracts 21 landmark points  
- Saves landmark vectors into:  

```
two_hand_dataset/<GESTURE_NAME>/
```

### ✅ Example Directory
```
two_hand_dataset/
    HELLO/
    HELP/
    NAME/
    THANK YOU/
    BYE/
```

Press **q** to close the webcam.

---

# 4️⃣ Train the Model

Once dataset is ready, train the neural network model:

```bash
cd src
python train_model.py
```

### ✅ Training Output:
Models will be saved in `models/`:

- `sign_model_best.keras`  
- `sign_model_final.keras`  
- Training logs saved to `training_log.csv`  

---

# 5️⃣ Convert Model to TFLite (Optional but Recommended)

For fast real-time performance:

```bash
python models/convert_to_tflite.py
```

Outputs:

- `sign_model_float32.tflite`
- `sign_model_int8.tflite`

---

# 6️⃣ Run Real-Time Web App

Navigate to the webapp folder:

```bash
cd webapp
python app.py
```

Open your browser:

👉 http://127.0.0.1:5000/

### ✅ Features of Web App:
- Real-time webcam detection  
- Progress bar (confidence meter)  
- Smooth UI  
- Uses TFLite model for fast performance  

---

# 7️⃣ Test Gesture Detection Without Web App

To test with OpenCV only:

```bash
python test_hands.py
```

---

# ✅ Overall Pipeline Summary

| Step | Script | Output |
|------|--------|--------|
| Data Collection | `collect_data.py` | Dataset stored in `two_hand_dataset/` |
| Training | `train_model.py` | `.keras` models saved in `/models/` |
| Convert Model | `convert_to_tflite.py` | `.tflite` models |
| Run Web App | `app.py` | Real-time gesture detection |

---

# 📸 Screenshots (Add Yours)

Place your UI and model screenshots in:

```
/screenshots
    ├── ui.png
    ├── prediction.png
    ├── dataset_example.png
    ├── training_plot.png
```

You can embed them like:

```markdown
![Web UI](screenshots/ui.png)
```

---

# ✅ Requirements

All required packages are listed in:

```
requirements.txt
```

Includes:

- TensorFlow
- MediaPipe
- OpenCV
- Flask
- NumPy
- Pandas

---

# ✅ License  
This project is licensed under the **MIT License**.

---

# 🙌 Author  
**Vikas Suthar**  
AIML Student | Deep Learning | Computer Vision  

