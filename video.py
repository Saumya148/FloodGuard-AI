import cv2
import os

video_path = "videos/t2.mp4"  # change to your video name
output_folder = "dataset/flood"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = len(os.listdir(output_folder))  # continue numbering

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save every 15th frame (IMPORTANT to avoid duplicates)
    if frame_count % 15 == 0:
        filename = f"flood_video_{saved_count}.jpg"
        cv2.imwrite(os.path.join(output_folder, filename), frame)
        saved_count += 1

    frame_count += 1

cap.release()

print("Frames Added:", saved_count)
