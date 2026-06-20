# Road Sign Detection System

A computer vision system that detects and classifies road signs in real time, built by training a YOLOv8m object detection model on road sign imagery.

## What it does

- Detects and classifies road signs from images using a custom-trained **YOLOv8m** model.
- Achieves **90% detection accuracy** on the trained dataset.
- Runs inference through a Python script for real-time/near-real-time detection.

## Tools & Stack

- **Python**
- **YOLOv8m** (Ultralytics)
- **Google Colab** (model training)
- OpenCV (image handling/inference pipeline)

## How it works

1. A custom road sign image dataset was used to fine-tune the YOLOv8m model on Google Colab.
2. The trained model weights are loaded by a Python inference script.
3. The script runs detection on input images, drawing bounding boxes and classification labels around identified road signs.

## Files

- `acw2.py` — main inference/detection script
- `code.zip` — full project code
- `Acw2 report.pdf` — full technical write-up and evaluation

## Status

Individual project. Trained and tested on a custom road sign dataset, 90% detection accuracy achieved.
