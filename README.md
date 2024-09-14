![final logo](https://github.com/user-attachments/assets/35b13121-ccc0-4514-9787-382764df5da6)

## Project Team:
  * Naif Almutairi
  * Fahad Alsaadi
  * Ali Alfares
  * Abdulaziz Alkathiri

## Description
This Streamlit application is designed to detect faces and estimate ages from video footage of persons entering and exiting a specific area. The application utilizes YOLOv5 for object detection, OpenCV for face detection, and a pre-trained age estimation model to identify and compare faces across two videos.

## Features
* Face Detection: Detect faces in video footage using OpenCV's pre-trained model.
* Age Estimation: Estimate the age of detected faces using a deep learning model.
* Video Processing: Analyze videos to detect and compare faces of children and adults.
* Abduction Alert: Alerts and highlights potential cases of child abduction based on face comparison.
* Sound Alert: Plays an alarm sound in case of suspected abduction.

## Requirments
* Python 3
* Streamlit
* OpenCV
* Torch
*NumPy
* scikit-image
* pygame
* playsound
  
## Usage
### Upload Videos:

* Upload the video file for the enter gate and exit gate.
* The app will process the videos and display results based on the detected faces and estimated ages.
### View Results:
* The app will display:
 * The processed frames with detected faces.
 * Comparison of face images between entering and exiting videos.
 * Alerts if a potential abduction is detected.
 * Play an alarm sound if the child is suspected to be abducted.

## Configuration
* Model Files: Place the model files in the modelNweight directory.
* Sound File: Ensure the alarm (1).mp3 file is in the root directory of the project for the alarm sound.

## file source
https://drive.google.com/drive/folders/1ajMZhRp02-xmgCAGhJXYH05PVF87n6o3?usp=sharing
