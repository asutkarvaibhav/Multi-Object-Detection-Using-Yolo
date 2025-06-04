# Multi-Object-Detection-Using-Yolo

# 🧿 Multi-Object Detection Using YOLO

This project implements real-time multi-object detection using the YOLO (You Only Look Once) algorithm. It can detect multiple objects in an image or video with high accuracy and speed.

---

## 🧰 Project Highlights

- Detects multiple objects in a single image/frame.
- Built using **YOLOv5 / YOLOv8** (depending on version used).
- Trained and evaluated on custom or standard datasets (e.g., COCO, VOC, or custom labeled images).
- Includes annotated datasets, training scripts, and model evaluation.

---

## 📂 Project Structure

```bash
├── data/                   # Images and annotation files
├── runs/                  # YOLO output: results and weights
├── models/                # Model config (optional)
├── detect.py              # Inference script
├── train.py               # Training script
├── requirements.txt       # Dependencies
├── README.md              # Project overview
└── results/               # Sample output images/videos


## 🚀 Getting Started

Follow these steps to set up and run the project:

1. **Clone the repository**

2. **Install required libraries**
Make sure you have Python 3.7+ installed, then run:

3. **Prepare the dataset**
- Place training and validation images in a directory (e.g., `data/images/`).
- Make sure annotations are in YOLO format inside `data/labels/`.
- Update the `data.yaml` file with the correct paths and class names.

4. **Train the YOLO model**
Example command to start training:

5. **Run inference on test images or videos**

6. **View results**
- Output images and predictions will be saved in the `runs/detect/` folder.

📊 Results
* Accuracy: ~XX% (replace with your mAP or precision if known)
* Sample detection:
<img src="results/sample_output.jpg" width="600"/>

⚙️ Tools & Frameworks
* Python 🐍
* YOLOv5 / YOLOv8
* OpenCV
* PyTorch
*LabelImg (for annotation)

**Created by Vaibhav Asutkar**


