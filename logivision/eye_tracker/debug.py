from logivision.eye_tracker.capturer import HaarCascadeBlobCapture
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av


def draw(img, point, color=(0, 255, 0), radius=5, thickness=5):
    cv2.circle(img, point, radius, color, thickness)


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # add mock points
    img[90, 300, :] = [0, 0, 255]
    img[90, 350, :] = [0, 0, 255]
    return av.VideoFrame.from_ndarray(img, format="bgr24")


def run():
    webrtc_streamer(key="sample", video_frame_callback=video_frame_callback)


if __name__ == "__main__":
    run()
