import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# 1. Load Model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

st.title("📹 Real-Time Person Counter")
st.write("Turn on the switch below to start the camera.")

# 2. Define the Processing Logic
# This function runs on every single video frame automatically
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Run YOLO detection
    results = model(img)
    
    person_count = 0
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if class_id == 0 and confidence > 0.5:
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
    # Draw the count directly on the video (easier for web streaming)
    cv2.putText(img, f"Count: {person_count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. Create the WebRTC Streamer
# This replaces the 'while' loop and 'cv2.VideoCapture'
webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
