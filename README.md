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


🚀 Getting Started
Clone the Repository

bash
Copy
Edit
git clone https://github.com/asutkarvaibhav/Multi-Object-Detection-Using-Yolo.git
cd Multi-Object-Detection-Using-Yolo
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Download/Prepare Dataset

Place your images and corresponding labels in the data/ directory.

If using a custom dataset, ensure it follows YOLO annotation format.

Train the Model

bash
Copy
Edit
python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt
Run Inference

bash
Copy
Edit
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --source data/test/
