import asyncio
import cv2
import numpy as np
from datetime import datetime

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# 🔥 TFLITE MODEL (KEPT FOR DEBUG ONLY)
from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter(
    model_path="flood_model.tflite",
    num_threads=4
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ INIT
app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ CAMERA
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# -----------------------------
# 🎯 FRONTEND
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -----------------------------
# 🎥 VIDEO STREAM
# -----------------------------
def generate_frames():
    while True:
        success, frame = cap.read()

        if not success:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Error", (50, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        frame = cv2.resize(frame, (320, 240))

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')

@app.get("/api/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# -----------------------------
# 🧠 PREPROCESS
# -----------------------------
def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def tflite_predict(img):
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# -----------------------------
# 📡 WEBSOCKET
# -----------------------------
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    prev_gray = None
    prev_velocity = 0
    frame_count = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                await asyncio.sleep(0.1)
                continue

            frame = cv2.resize(frame, (160,120))
            frame_count += 1

            # 🔥 MODEL RUN (DEBUG ONLY)
            if frame_count % 5 == 0:
                img = preprocess(frame)
                pred = tflite_predict(img)

                if len(pred[0]) == 2:
                    prediction = float(pred[0][1])
                else:
                    prediction = float(pred[0][0])

                print("MODEL OUTPUT:", prediction)

            # -----------------------------
            # 🔹 VELOCITY
            # -----------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            velocity = 0.0

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                velocity = float(np.mean(diff))

                if velocity < 5:
                    velocity = 0

            prev_gray = gray

            velocity = 0.8 * prev_velocity + 0.2 * velocity
            prev_velocity = velocity

            if velocity > 100:
                velocity = 100

            # -----------------------------
            # 🔹 WATER DETECTION (FIXED)
            # -----------------------------
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower = np.array([10, 50, 50])
            upper = np.array([30, 255, 200])

            mask = cv2.inRange(hsv, lower, upper)
            water_ratio = np.sum(mask > 0) / mask.size

            if water_ratio < 0.02:
                water_ratio = 0

            # -----------------------------
            # 🔥 FINAL DECISION (MATCH LAPTOP)
            # -----------------------------
            if water_ratio > 0.05 and velocity > 20:
                status = "FLOOD"
            elif water_ratio > 0.05:
                status = "WARNING"
            else:
                status = "SAFE"

            # DEBUG PRINT
            print(
                "| Vel:", round(velocity, 2),
                "| Water:", round(water_ratio, 2),
                "| Status:", status
            )

            await websocket.send_json({
                "velocity": round(velocity, 2),
                "status": status,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

            await asyncio.sleep(0.8)

    except Exception as e:
        print("WebSocket Error:", e)

# -----------------------------
# 🛑 SHUTDOWN
# -----------------------------
@app.on_event("shutdown")
def shutdown_event():
    cap.release()