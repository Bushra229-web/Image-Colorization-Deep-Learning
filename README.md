# Image-Colorization-Deep-Learning
Image colorization using convolutional autoencoders (MPhil AI Project)
Overview

This project implements two deep learning computer vision tasks:

Image Colorization using a Convolutional Autoencoder

Converts grayscale images into realistic RGB images.

Object Detection and Multi-Object Tracking

Detects and tracks multiple objects in videos using YOLOv8 and DeepSORT.

The goal is to demonstrate image reconstruction, image-to-image translation, and real-time detection/tracking.
Methodology
1️⃣ Autoencoder (Colorization)

Converted RGB images to grayscale as input.

Built a Convolutional Autoencoder with encoder-decoder layers.

Trained on 500 images resized to 128×128.

Evaluated using MSE and PSNR.

Visualized results comparing grayscale input, original RGB, and predicted RGB.

2️⃣ Object Detection & Tracking

Loaded short traffic video.

Performed real-time object detection using YOLOv8 pretrained weights.

Applied DeepSORT for multi-object tracking.
Dataset
Autoencoder

Source: Coin Dataset subset (Google Drive)

Number of images: 500

Resolution: 128×128 pixels

Object Detection

Short video clips (traffic/pedestrians) used for demonstration.

⚠ Dataset and large model files are not included. Use Google Drive link if needed.
Training Details
Autoencoder

Epochs: 20

Batch size: 16

Optimizer: Adam

Loss Function: MSE

Object Detection

Pretrained YOLOv8 weights used.

DeepSORT tracks objects using Kalman Filter and Hungarian Algorithm.
Results
Autoencoder

Validation MSE: 0.003

Average PSNR:24.89
Tech Stack

Python 3

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

Ultralytics YOLOv8

DeepSORT

Strengths

Efficient autoencoder for colorization

Real-time YOLOv8 detection

Stable multi-object tracking using DeepSORT

Modular and easy-to-run notebooks
Limitations

Slight blurring in colorized images

Small dataset size (500 images)

Tracking may fail under heavy occlusion

Detection uses pretrained weights; fine-tuning may improve results

Author

Bushra Tasadaq
Master’s in Computer Science
