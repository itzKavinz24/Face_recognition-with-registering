# Real-Time Face Recognition System

A Python-based real-time face recognition system using OpenCV. This project allows you to register faces into a dataset and later recognize them through a live webcam feed using the FisherFace algorithm.

## 🔍 Features

- Face detection using Haar Cascade
- Face registration and dataset creation
- FisherFace algorithm for training and recognition
- Real-time recognition via webcam
- Unknown face detection and logging

## 🛠 Technologies Used

- Python
- OpenCV (cv2)
- NumPy

## 📂 Project Structure

├── dataset/ # Stored face images (created during registration)
├── haarcascade_frontalface_default.xml # Haar Cascade XML file
├── register_faces.py # Script to register and save face images
├── recognise_face.py # Script to train and recognize faces via webcam


## 📸 How It Works

### 1. Face Registration

Run the script to capture and store face images:

```bash
python register_faces.py

    A new folder is created in dataset/ with the person's name.

    Captures multiple face images per person.

2. Recognition

Run the recognition script:

python recognise_face.py

    Trains the model using stored dataset images.

    Uses webcam to detect and recognize faces in real-time.

    Displays name and confidence score on screen.

    Logs "Unknown" if the face is not recognized.

✅ Requirements

    Python 3.x

    OpenCV (pip install opencv-contrib-python)

    NumPy
