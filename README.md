# 🤟 American Sign Language (ASL) Recognizer

Welcome to the **ASL Recognizer**! This project utilizes computer vision and deep learning to detect and classify American Sign Language (ASL) hand gestures in real-time. It includes custom scripts for collecting training images, preprocessing them to a standardized format, and performing inference using a pre-trained Keras model.

---

> 🎓 **Note:** This is my third project in **Computer Vision**! I built it to gain hands-on experience with dataset collection, image normalization, deep learning model deployment, and real-time classification.

---

## 🚀 Features

*   **Real-time Gesture Recognition:** Detects ASL alphabets and numbers using a camera feed and displays labels on screen.
*   **Standardized Data Collection:** Automatically crops, padding-adjusts, and resizes hand regions of interest (ROI) to a uniform `300x300` canvas while keeping the original aspect ratio.
*   **Deep Learning Inference:** Integrates a pre-trained Keras neural network model (`.h5`) for gesture classification via `cvzone`'s Classifier module.
*   **Live Prediction & Bounding Boxes:** Draws dynamic bounding boxes and displays the predicted character classification directly above the detected hand.

---

## 🛠️ Prerequisites & Installation

### 1. Requirements
*   Python 3.8 - 3.10 is recommended (ensure TensorFlow compatibility).
*   A webcam connected to your system.

### 2. Install Dependencies
Install the required libraries:
```bash
pip install opencv-python cvzone numpy tensorflow
```

---

## 📁 Project Structure

```text
ASL/
├── Model/
│   ├── keras_model.h5       # Pre-trained Keras classification model
│   └── labels.txt           # File containing classification class labels
├── Data Collection.py      # Script used to capture and save training datasets
├── Testing.py              # Script for running live webcam inference and classification
└── README.md               # Project documentation
```

---

## 💻 How to Use

### Phase 1: Data Collection (Optional)
If you want to train your own model or gather data:
1.  Open `Data Collection.py` and modify the `path` variable (e.g., `Data/A`, `Data/B`, etc.) to match the folder where you want to save images.
2.  Run the script:
    ```bash
    python "Data Collection.py"
    ```
3.  Place your hand in front of the camera and perform the sign.
4.  Press the **`s`** key to save the preprocessed hand frame to the target directory.

### Phase 2: Running Inference (Testing)
To run the real-time recognition using the trained model:
1.  Ensure your model files (`keras_model.h5` and `labels.txt`) are placed in the `Model/` directory.
2.  Run the script:
    ```bash
    python Testing.py
    ```
3.  The webcam feed will launch. Place your hand in the frame, and the system will detect and classify the ASL letter or number, drawing a bounding box and showing the label.
4.  Press **`q`** or close the window to exit.

---

## 🧠 Skills & Concepts Learned

By building this project, I gained practical experience in:

*   **Dataset Pipeline Design:** Structuring a data gathering script to compile custom training sets for computer vision models.
*   **Image Standardization & Normalization:**
    *   Cropping custom bounding boxes dynamically.
    *   Preserving hand aspect ratios when scaling and fitting raw crops into a fixed `300x300` white canvas.
    *   Centering images using mathematical calculations (`wGap` and `hGap`).
*   **Deep Learning Deployment:** Deploying pre-trained Keras (`.h5`) models locally and executing low-latency real-time inference on a video stream.
*   **Hand Tracking and Segmentation:** Masking, cropping, and isolating hand coordinates from background noise using `cvzone`.

---

## 🛠️ Tech Stack & Libraries

This project applies several core technologies and packages:

*   **Python:** Programming language.
*   **OpenCV (cv2):** Used for video capturing, bounding box rendering, text rendering, cropping, resizing, and saving images.
*   **cvzone (HandTracking & Classification):** Used for hand detection and loaded as a lightweight wrapper to execute TensorFlow/Keras model predictions.
*   **TensorFlow/Keras:** The deep learning framework hosting the `.h5` model architecture and weights.
*   **NumPy:** Used for generating canvas matrices and pixel additions.
