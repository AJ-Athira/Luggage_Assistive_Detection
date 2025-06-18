# 🎒 Smart Luggage Detection System

This is a computer vision prototype using **YOLOv8**, **OpenCV**, and **Text-to-Speech (pyttsx3)** to detect various types of luggage such as suitcases, backpacks, handbags, and briefcases in real-time or from a recorded video. It provides **visual feedback** via bounding boxes and **auditory feedback** via speech.

---

## 🧠 Features

- Detects **multiple types of luggage**: suitcase, backpack, handbag, briefcase.
- Supports both **live webcam feed** and **recorded video files**.
- Resizes output window to **fit most screens** for better visualization.
- Provides **audio alerts** when luggage is detected using `pyttsx3`.
- Uses **YOLOv8n** (nano) model for fast, lightweight performance.



## 📦 Requirements

Install the required dependencies:

```bash
pip install ultralytics opencv-python pyttsx3
````

Also make sure you have the YOLOv8 model file:

```bash
# Download YOLOv8n model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

> Or use `from ultralytics import YOLO; model = YOLO('yolov8n.pt')` and it will auto-download.



## 🚀 How to Run

Run the Python script:

```bash
python luggage_detection.py
```

Then, choose:

* `"live"` for **live webcam detection**
* `"file"` for **recorded video detection** — you will be prompted to enter the file path (e.g., `luggage.mp4`)



## 🧩 Supported Labels

* Suitcase
* Backpack
* Handbag
* Briefcase

These are part of the **COCO dataset labels** that YOLO is trained on.


## 🔊 Audio Feedback

When a new luggage item is detected, the system will speak out the label (e.g., *"backpack detected"*) using `pyttsx3`. Each label will be announced only once every few seconds to avoid repetition.



## 🛠️ Future Work

* Add object tracking for smoother feedback.
* Integrate with hardware for a wearable solution (e.g., Raspberry Pi + buzzer/vibration).
* Improve detection accuracy with a fine-tuned model.



## 🧠 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [OpenCV](https://opencv.org/)
* [pyttsx3](https://pypi.org/project/pyttsx3/)
