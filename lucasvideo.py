import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import winsound   # Comment if not Windows

# ==============================
# LOAD MODEL
# ==============================
model = tf.keras.models.load_model("flood_model.h5")
IMG_SIZE = 224

# ==============================
# PIXEL TO METER CONVERSION
# ==============================
PIXEL_TO_METER = 0.02   # Adjust based on your camera calibration

# ==============================
# VIDEO
# ==============================
cap = cv2.VideoCapture("h.mp4")

if not cap.isOpened():
    print("Error opening video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)

# ==============================
# LUCAS–KANADE SETTINGS
# ==============================
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

feature_params = dict(
    maxCorners=40,
    qualityLevel=0.4,
    minDistance=15,
    blockSize=7
)

# Read first frame
ret, old_frame = cap.read()
if not ret:
    print("Error reading first frame")
    exit()

frame_height, frame_width = old_frame.shape[:2]

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_gray = cv2.GaussianBlur(old_gray, (5, 5), 0)

# Mask lower region (water focus)
mask = np.zeros_like(old_gray)
mask[int(frame_height * 0.4):frame_height, :] = 255

p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)

# ==============================
# FIX WINDOW SCALING
# ==============================
cv2.namedWindow("Flood Monitoring System", cv2.WINDOW_NORMAL)

screen_width = 1200
scale_ratio = screen_width / frame_width
new_width = int(frame_width * scale_ratio)
new_height = int(frame_height * scale_ratio)

cv2.resizeWindow("Flood Monitoring System", new_width, new_height)

# Velocity smoothing
velocity_buffer = []

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # ==============================
    # CNN FLOOD PREDICTION
    # ==============================
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)
    prob = prediction[0][0]

    if prob > 0.5:
        flood_label = "NON FLOOD"
    else:
        flood_label = "FLOOD"

    # ==============================
    # LUCAS–KANADE MOTION
    # ==============================
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    else:
        good_new = []
        good_old = []

    motion = 0

    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        motion += np.sqrt((a - c)**2 + (b - d)**2)
        cv2.circle(frame, (int(a), int(b)), 2, (0, 255, 0), -1)

    avg_motion = motion / len(good_new) if len(good_new) > 0 else 0

    # ==============================
    # SMOOTH VELOCITY (px/sec → m/sec)
    # ==============================
    velocity_buffer.append(avg_motion * fps)

    if len(velocity_buffer) > 10:
        velocity_buffer.pop(0)

    velocity_px = sum(velocity_buffer) / len(velocity_buffer)
    velocity = velocity_px * PIXEL_TO_METER   # Convert to meter/sec

    # ==============================
    # INTELLIGENT SEVERITY SYSTEM (Adjusted for m/sec)
    # ==============================
    if flood_label == "NON FLOOD":
        severity = "NO RISK"
        status = "SAFE"

    else:
        if velocity < 0.8:           # ~40 px/sec equivalent
            severity = "LOW RISK"
            status = "STAGNANT FLOOD"

        elif 0.8 <= velocity <= 1.6:  # ~40–80 px/sec equivalent
            severity = "MEDIUM RISK"
            status = "MODERATE FLOW"

        else:
            severity = "HIGH RISK"
            status = "ACTIVE FLOOD FLOW"

    # ==============================
    # ALERT ONLY FOR HIGH RISK
    # ==============================
    if severity == "HIGH RISK":
        try:
            winsound.Beep(1000, 200)
        except:
            pass

    # ==============================
    # LOGGING
    # ==============================
    with open("flood_log.txt", "a") as f:
        f.write(f"{datetime.now()} | {status} | Velocity: {velocity:.2f} m/sec\n")

    # ==============================
    # DISPLAY TEXT
    # ==============================
    cv2.putText(frame, f"Flood: {flood_label}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Velocity: {velocity:.2f} m/sec", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame, f"Severity: {severity}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, f"Status: {status}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Flood Monitoring System", frame)

    # Update previous frame
    old_gray = gray.copy()

    if len(good_new) > 0:
        p0 = good_new.reshape(-1, 1, 2)
    else:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
