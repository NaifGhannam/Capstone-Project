import streamlit as st
import cv2
import torch
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from playsound import playsound
import base64
import pygame

# Initialize the pygame mixer

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load face detection and age estimation models
faceProto = "modelNweight/opencv_face_detector.pbtxt"
faceModel = "modelNweight/opencv_face_detector_uint8.pb"
ageProto = "modelNweight/age_deploy.prototxt"
ageModel = "modelNweight/age_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

ageNet = cv2.dnn.readNet(ageModel, ageProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Function to detect face bounding boxes
def getFaceBox(net, frame, conf_threshold=0.7):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (600, 600), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frameWidth, x2), min(frameHeight, y2)
            bboxes.append([x1, y1, x2, y2])
    return bboxes

# Function to detect age
def detect_age(face):
    if face.size == 0:
        raise ValueError("Empty face image provided for age detection")
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    return ageList[agePreds[0].argmax()]

# Function to process video and detect faces/age
def process_video(video_path, save_dir, frame_tag):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(save_dir, exist_ok=True)
    cropped_faces_dict = {'child': []}
    frame_count = 0
    frame_saved = False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    progress_bar = st.progress(0)  # Initialize the progress bar

    while cap.isOpened() and not frame_saved:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Update progress bar based on the number of frames processed
        progress = int((frame_count / total_frames) * 100)
        progress_bar.progress(progress)

        # Perform object detection with YOLOv5
        results = model(frame)
        predictions = results.pandas().xyxy[0]

        # Filter detections for persons
        person_detections = predictions[predictions['name'] == 'person']

        # List to store detected face images in the current frame
        faces_in_frame = []

        for idx, row in person_detections.iterrows():
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            confidence = row['confidence']

            # Only consider persons with confidence greater than 0.8
            if confidence > 0.8:
                # Crop the person region from the frame
                person_region = frame[y1:y2, x1:x2]

                # Detect faces within the person region
                face_bboxes = getFaceBox(faceNet, person_region)

                for bbox in face_bboxes:
                    # Adjust bounding box to original frame coordinates
                    x1_face = max(0, bbox[0] + x1)
                    y1_face = max(0, bbox[1] + y1)
                    x2_face = min(frame.shape[1], bbox[2] + x1)
                    y2_face = min(frame.shape[0], bbox[3] + y1)

                    # Crop the face
                    face = frame[y1_face:y2_face, x1_face:x2_face]

                    if face.size > 0:
                        # Add face to list of faces in the current frame
                        faces_in_frame.append(face)

                        # Detect age of the face
                        try:
                            age = detect_age(face)

                            if age in ['(0-2)', '(4-6)', '(8-12)']:
                                
                                if len(faces_in_frame) == 2:
                                    save_frame_path = os.path.join(save_dir, f'{frame_tag}_frame_with_two_faces_{frame_count}.jpg')
                                    cv2.imwrite(save_frame_path, frame)
                                    st.write(f"Child detected")
                                    cropped_faces_dict['child'] = faces_in_frame
                                    frame_saved = True
                                    break
                        except Exception as e:
                            st.write(f"Error detecting age: {e}")

        if frame_saved:
            break

    cap.release()
    progress_bar.empty()  # Remove the progress bar after processing is complete
    return cropped_faces_dict

# Function to calculate image similarity
def calculate_similarity(img1, img2):
    # Resize images to be the same size
    img1_resized = cv2.resize(img1, (256, 256))
    img2_resized = cv2.resize(img2, (256, 256))
    
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score

# Function to compare faces between two sets of images
# Function to compare faces between two sets of images and display comparisons
def compare_first_images(images_in, images_out):
    if not images_in or not images_out:
        st.write("One of the image lists is empty.")
        return False

    img_in = images_in[1]
    img_out = images_out[1]

    # Display the first pair of images for comparison
    st.write("Comparing the first pair of images:")
    st.image([img_in, img_out], caption=["Entering Child", "Exiting Child"], width=300)

    # Calculate similarity
    similarity = calculate_similarity(img_in, img_out)
    threshold = 0.6  

    if similarity > threshold:
        st.write("The child seems to be exiting.")
        
        # Display the similarity score

        if len(images_in) > 1 and len(images_out) > 1:
            img_in = images_in[0]
            img_out = images_out[0]

            # Display the second pair of images for comparison
            st.write("Comparing the second pair of images:")
            st.image([img_in, img_out], caption=["Entering person", "Exiting person"], width=300)

            try:
                similarity = calculate_similarity(img_in, img_out)

                if similarity > threshold:
                    st.success("The child is with their parent.")
                else:
                    st.warning("⚠️ Warning: The child might be abducted.")
                    display_abduction_images(images_out)
                    pygame.mixer.init()

                    # Load the sound file
                    pygame.mixer.music.load("alarm (1).mp3")

                    # Start playing the sound (auto-play)
                    pygame.mixer.music.play()

                    # Keep the program running until the sound finishes playing
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)

                                        
                    
                    
                    
                    
                    # # Play the alarm sound if there's a potential abduction
                    # with open("alarm (1).mp3", "rb") as f:
                    #     data = f.read()
                    #     b64 = base64.b64encode(data).decode()
                    #     audio_html = f"""
                    #         <audio autoplay controls>
                    #         <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    #         </audio>
                    #         """

                    # # Streamlit interface to trigger sound
                    # st.write("# Auto-playing Audio!")

                    # if st.button("Play Sound", key="sound_button"):
                    #     st.markdown(audio_html, unsafe_allow_html=True)

                    # # JavaScript to automatically "click" the button after the page loads
                    # st.markdown("""
                    #     <script>
                    #     // Automatically click the button after the page loads
                    #     document.querySelector('button[title="Play Sound"]').click();
                    #     </script>
                    #     """, unsafe_allow_html=True)

            except Exception as e:
                st.write(f"Error comparing the second set of images: {e}")
        else:
            st.write("Not enough images for comparison.")
    else:
        st.write("No child detected exiting.")

# Function to display abduction images in Streamlit
def display_abduction_images(images_out):
    st.write("Displaying suspected abduction images:")
    if images_out:
        for i, img in enumerate(images_out):
            st.image(img, caption=f"Abduction Image {i + 1}")
    else:
        st.write("No images available for display.")

# Streamlit interface
st.image("final_logo.png")

# Upload two videos
video_file_1 = st.file_uploader("Upload the enter gate video", type=["mp4", "avi", "mov"])
video_file_2 = st.file_uploader("Upload the exit gate video", type=["mp4", "avi", "mov"])

if video_file_1 is not None:
    st.write("Playing the enter gate video:")
    st.video(video_file_1)  # Play the first video
    
if video_file_2 is not None:
    st.write("Playing the exit gate video:")
    st.video(video_file_2)  # Play the second video

if video_file_1 is not None and video_file_2 is not None:
    st.write("Processing both videos, please wait...")

    # Save the uploaded videos to a temporary folder
    temp_dir = "temp_videos"
    os.makedirs(temp_dir, exist_ok=True)

    video_path_1 = os.path.join(temp_dir, "video_1.mp4")
    video_path_2 = os.path.join(temp_dir, "video_2.mp4")

    with open(video_path_1, mode='wb') as f:
        f.write(video_file_1.read())
    with open(video_path_2, mode='wb') as f:
        f.write(video_file_2.read())

    # Process both videos
    cropped_objects_dict_in = process_video(video_path_1, "saved_faces_in", "video_1")
    cropped_objects_dict_out = process_video(video_path_2, "saved_faces_out", "video_2")

    # Display results
    st.success("✅ Processing completed")
        
        # Compare faces between the two videos
    compare_first_images(cropped_objects_dict_in['child'], cropped_objects_dict_out['child'])
