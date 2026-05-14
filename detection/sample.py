import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

 
    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)

           
            label = f"{model.names[int(cls)]} ({score:.2f})"

           
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            
            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    return av.VideoFrame.from_ndarray(img, format="bgr24")



webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    media_stream_constraints={"video": True, "audio": False},
)