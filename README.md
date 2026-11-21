\# PPE Violation Detection System



\## Overview

A real-time computer vision application designed to detect Personal Protective Equipment (PPE) violations at construction sites. The system uses an AI object detection model to identify workers, hardhats, and safety vests, helping improve workplace safety through automated monitoring.



\## Features

\- Real-time PPE detection using a TFLite model

\- Detects workers, hardhats, and safety vests

\- Identifies PPE violations

\- Captures violation images automatically

\- Saves violation events locally

\- Real-time camera inference

\- Optimized for edge AI deployment



\## Detection Classes

| Class | Description |

|---|---|

| Person | Detects workers |

| Hardhat | Detects safety helmets |

| Safety Vest | Detects safety vests |



\## System Architecture



Camera Input → Frame Capture → Image Preprocessing → AI Inference → Object Detection → PPE Violation Analysis → Alert \& Image Capture



\## AI Model

\- Model format: TensorFlow Lite

\- Model input resolution: 640 × 640

\- Confidence threshold: 0.40

\- IoU threshold: 0.45

\- Number of inference threads: 4



The system uses a trained object detection model to identify workers and their protective equipment.



\## Project Structure



```text

PPE-Violation-Detection/

│

├── 1\_ppe\_runtime.py

├── ppe\_detection.py

├── ppe\_image\_detection.py

├── ppe\_realtime.py

├── camera\_inference.py

├── debug\_boxes.py

├── test\_model.py

├── Backup.py

│

├── best.tflite

├── best.onnx

│

├── construction-worker-site.jpg

│

└── README.md

```



\## Technologies Used

\- Python

\- OpenCV

\- NumPy

\- TensorFlow Lite

\- ONNX

\- Computer Vision

\- Edge AI

\- GStreamer



\## Deployment Platform

Developed for edge AI deployment, including Qualcomm-based embedded Linux platforms.



The runtime implementation uses a GStreamer camera pipeline and a TensorFlow Lite inference model.



\## Applications

\- Construction site safety monitoring

\- Industrial workplace monitoring

\- PPE compliance detection

\- Smart surveillance systems

\- Edge AI safety applications



\## Future Improvements

\- Telegram notifications for PPE violations

\- Web-based monitoring dashboard

\- Worker tracking

\- Violation statistics and reporting

\- Multi-camera support

\- Edge AI hardware acceleration



\## Disclaimer

This project is intended for demonstration and research purposes. It should not replace certified workplace safety procedures or human supervision.



