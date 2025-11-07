import numpy as np
import cv2
from PIL import Image

def load_video(video_path:str, top_frame_num=5):
    # Read file
    cap = cv2.VideoCapture(video_path)
    output_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if len(output_frames) > top_frame_num:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        output_frames.append(np.array(pil_img))

    return output_frames

if __name__ == "__main__":
    video_path = "/mnt/xiping/spinning-earth-480.mp4"
    frames = load_video(video_path=video_path)
    print(np.shape(np.array(frames)))