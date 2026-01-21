import streamlit as st
import cv2
from ultralytics import YOLO

# 1. Load Model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# 2. Set up the Page
st.title("📹 Real-Time Person Counter")

col1, col2 = st.columns([3, 1])

with col2:
    run_detection = st.checkbox('Start Camera', value=True)
    kpi_text = st.markdown("### Person Count: 0")
    conf_threshold = st.slider("Confidence", 0.0, 1.0, 0.5, 0.05)

with col1:
    # --- CRITICAL STEP 1 ---
    # We create an EMPTY box (placeholder) here, OUTSIDE the loop.
    # We name it 'frame_window'.
    frame_window = st.image([])

# 3. Video Logic
cap = cv2.VideoCapture(0)

while run_detection:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video.")
        break

    # Run YOLOv8 inference
    results = model(frame)

    # Extract detections
    person_count = 0
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id == 0 and confidence > conf_threshold:
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"Person: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Convert BGR (OpenCV) to RGB (Streamlit)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- CRITICAL STEP 2 (THE FIX) ---
    # We talk to 'frame_window' (the box we made earlier).
    # We do NOT use 'st.image()' here.
    frame_window.image(frame_rgb)
    
    # Update count
    kpi_text.write(f"### Person Count: {person_count}")

cap.release()