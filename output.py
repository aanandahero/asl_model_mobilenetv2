import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque, Counter
import time
import streamlit as st

# Load the trained model
model = tf.keras.models.load_model(r"asl_transfer_mobilenetv2.h5")

# Define class labels (A-Z, Space, Delete, Nothing)
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8)

# Streamlit App
st.title("ASL Sign Language Detection")
st.write(" Turn on your webcam to detect ASL signs")

# Initialize Streamlit elements
frame_placeholder = st.empty()
text_placeholder = st.empty()

# Button to start/stop webcam
run_webcam = st.button("Start Webcam")

def detect_asl():
    cap = cv2.VideoCapture(0)
    predicted_text = ""
    frame_buffer = deque(maxlen=15)
    prediction_delay = 0.5
    last_prediction_time = time.time()
    consecutive_frames = 5

    while run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min = max(0, int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]))
                y_min = max(0, int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]))
                x_max = min(frame.shape[1], int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]))
                y_max = min(frame.shape[0], int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]))

                # Crop hand region
                hand_roi = frame[y_min:y_max, x_min:x_max]

                if hand_roi.size == 0:
                    continue

                # Preprocess hand image
                img = cv2.resize(hand_roi, (128, 128))
                img = img / 255.0  # Normalize
                img = np.expand_dims(img, axis=0)

                # Make prediction
                predictions = model.predict(img)
                confidence = np.max(predictions)
                predicted_class = class_labels[np.argmax(predictions)]

                # If confidence is high, add to the buffer
                if confidence > 0.7:
                    frame_buffer.append(predicted_class)

                if len(frame_buffer) == frame_buffer.maxlen:
                    most_common_prediction = Counter(frame_buffer).most_common(1)[0][0]

                    if time.time() - last_prediction_time > prediction_delay:
                        if most_common_prediction == "space":
                            predicted_text += " "
                        elif most_common_prediction == "del":
                            predicted_text = predicted_text[:-1]
                        elif most_common_prediction != "nothing":
                            predicted_text += most_common_prediction

                        last_prediction_time = time.time()
                        frame_buffer.clear()

                # Display prediction on frame
                cv2.putText(frame, f"Prediction: {predicted_class} ({confidence:.2f})", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show accumulated text
        text_placeholder.write(f" Recognized Text: {predicted_text}")

        # Show frame in Streamlit
        frame_placeholder.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

if run_webcam:
    detect_asl()
    st.stop()  # Ensure Streamlit stops executing further