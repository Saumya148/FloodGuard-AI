import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from datetime import datetime
import os

# ==============================
# MAX PERFORMANCE SETTINGS
# ==============================
cv2.setUseOptimized(True)
cv2.setNumThreads(4)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

# ==============================
# LOAD MODEL
# ==============================
interpreter = Interpreter(
    model_path="flood_model.tflite",
    num_threads=4
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 224
PIXEL_TO_METER = 0.02
CNN_SKIP = 3          # 🔥 more skip
INV_255 = 1.0 / 255.0

input_buffer = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

# ==============================
# CAPTURE DIRECT SMALL RES
# ==============================
cap = cv2.VideoCapture("h.mp4")

# Force small capture resolution if camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

# ==============================
# LK SETTINGS (lighter)
# ==============================
lk_params = dict(
    winSize=(10, 10),
    maxLevel=1,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 6, 0.03)
)

feature_params = dict(
    maxCorners=20,      # 🔥 reduced
    qualityLevel=0.4,
    minDistance=10,
    blockSize=5
)

# ==============================
# INITIAL FRAME
# ==============================
ret, frame = cap.read()
if not ret:
    exit()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(gray)
h = gray.shape[0]
mask[int(h * 0.4):h, :] = 255

p0 = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)

cv2.namedWindow("Flood Monitoring System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Flood Monitoring System", 960, 720)

# ==============================
# FAST EMA
# ==============================
velocity = 0.0
ALPHA = 0.35

log_buffer = []
frame_counter = 0
flood_label = "UNKNOWN"

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ==============================
    # CNN (skipped more aggressively)
    # ==============================
    if frame_counter % CNN_SKIP == 0:
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        input_buffer[0] = resized * INV_255

        interpreter.set_tensor(input_details[0]['index'], input_buffer)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        prob = float(output[0][0])
        flood_label = "NON FLOOD" if prob > 0.5 else "FLOOD"

    frame_counter += 1

    # ==============================
    # OPTICAL FLOW
    # ==============================
    motion = 0.0
    count = 0

    if p0 is not None:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            gray, gray_new, p0, None, **lk_params
        )
    else:
        p1 = None

    if p1 is not None and st is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        count = good_new.shape[0]

        if count > 0:
            diff = good_new - good_old
            motion = np.sum(np.hypot(diff[:, 0], diff[:, 1]))
            p0 = good_new.reshape(-1, 1, 2)
        else:
            p0 = None
    else:
        p0 = None

    if p0 is None or (p0 is not None and p0.shape[0] < 8):
        p0 = cv2.goodFeaturesToTrack(gray_new, mask=mask, **feature_params)

    avg_motion = (motion / count) if count > 0 else 0.0

    # ==============================
    # FAST EMA SMOOTH
    # ==============================
    velocity_real = avg_motion * fps * PIXEL_TO_METER
    velocity = (ALPHA * velocity_real) + ((1 - ALPHA) * velocity)

    # ==============================
    # SEVERITY
    # ==============================
    if flood_label == "NON FLOOD":
        severity = "NO RISK"
        status = "SAFE"
    else:
        if velocity < 0.8:
            severity = "LOW RISK"
            status = "STAGNANT FLOOD"
        elif velocity <= 1.6:
            severity = "MEDIUM RISK"
            status = "MODERATE FLOW"
        else:
            severity = "HIGH RISK"
            status = "ACTIVE FLOOD FLOW"

    # ==============================
    # LOGGING (low frequency)
    # ==============================
    if frame_counter % 20 == 0:
        log_buffer.append(
            f"{datetime.now()} | {status} | Velocity: {velocity:.2f} m/sec\n"
        )

    if len(log_buffer) >= 40:
        with open("flood_log.txt", "a") as f:
            f.writelines(log_buffer)
        log_buffer.clear()

    # ==============================
    # DISPLAY (scaled up cheap)
    # ==============================
    display = cv2.resize(frame, (960, 720))

    cv2.putText(display, f"Flood: {flood_label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, f"Velocity: {velocity:.2f} m/sec", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(display, f"Severity: {severity}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(display, f"Status: {status}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Flood Monitoring System", display)

    gray = gray_new

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

if log_buffer:
    with open("flood_log.txt", "a") as f:
        f.writelines(log_buffer)

cv2.destroyAllWindows()