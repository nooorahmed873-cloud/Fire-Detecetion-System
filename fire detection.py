from flask import Flask, Response, render_template, request, redirect, url_for, jsonify, send_from_directory
import cv2
from ultralytics import YOLO
import time
import os
import numpy as np
import requests


app = Flask(__name__)



model = YOLO("best.pt")


# =========================
# 📲 TELEGRAM SETTINGS
# =========================

bot_token="YOUR_BOT_TOKEN"
chat_id="YOUR_CHAT_ID"


def send_telegram_alert():

    try:

        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

        data = {
            "chat_id": CHAT_ID,
            "text": "🔥 WARNING! Fire detected by AI system"
        }

        requests.post(url, data=data)

        print("Telegram alert sent")

    except Exception as e:

        print("Telegram error:", e)



# =========================
# 🎥 Video
# =========================

cap = None
latest_result = "false"
video_fps = 25


# =========================
# 🔔 Fire Status
# =========================

alarm_status = False
last_fire_time = 0
FIRE_TIMEOUT = 1.5

alert_sent = False


# =========================
# 📁 Folders
# =========================

os.makedirs("uploads", exist_ok=True)


# =========================
# 🤖 Detection
# =========================

def detect_fire(frame):

    global alarm_status
    global last_fire_time
    global alert_sent


    results = model.predict(
        frame,
        conf=0.25,
        verbose=False
    )


    fire_detected = len(results[0].boxes) > 0
    if fire_detected:
        print("🔥 FIRE DETECTED")


    if fire_detected:

        alarm_status = True
        last_fire_time = time.time()

        # 📲 send once only
        if not alert_sent:
            send_telegram_alert()
            alert_sent = True

        return "true"


    if time.time() - last_fire_time > FIRE_TIMEOUT:

        alarm_status = False
        alert_sent = False

    return "false"



# =========================
# 🎥 Stream Generator
# =========================

def generate_frames():

    global latest_result
    global cap
    global video_fps

    time.sleep(5)

    last_time = 0


    frame_count = 0
    while True:
        success,frame = cap.read()
        frame_count += 1
        if frame_count % 3 ==0:
            detect_fire(frame)

        if not success:
            break

        frame = cv2.resize(frame, (640, 480))

        if time.time() - last_time > 0.5:

            latest_result = detect_fire(frame)
            last_time = time.time()

        color = (0, 0, 255) if latest_result == "true" else (0, 255, 0)

        cv2.putText(
            frame,
            f"Fire: {latest_result}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

        time.sleep(1 / video_fps)



# =========================
# 🌐 Home
# =========================

@app.route("/", methods=["GET", "POST"])
def home():

    global cap, video_fps

    if request.method == "POST":

        # حفظ صوت الإنذار
        alarm_file = request.files.get("alarm")

        if alarm_file and alarm_file.filename:
            alarm_file.save("alarm.mp3")
            print("Alarm saved")

        # حفظ الفيديو
        file = request.files["video"]

        if file:

            path = os.path.join("uploads", file.filename)
            file.save(path)

            cap = cv2.VideoCapture(path)

            fps = cap.get(cv2.CAP_PROP_FPS)
            video_fps = fps if fps and fps > 0 else 25

            return redirect(url_for("video_page"))

    return render_template("index.html")
# =========================
# 🎥 Stream
# =========================

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )



@app.route("/play")
def video_page():
    return render_template("video.html")



# =========================
# 🔊 Alarm File
# =========================

@app.route("/alarm.mp3")
def alarm():
    return send_from_directory(
        os.getcwd(),
        "alarm.mp3",
        mimetype="audio/mpeg"
    )



# =========================
# 📡 Status API
# =========================

@app.route("/status")
def status():
    return jsonify({
        "fire": bool(alarm_status)
    })



@app.route("/api/fire")
def api_fire():
    return jsonify({
        "fire": bool(alarm_status)
    })



# =========================
# 📸 Image API
# =========================

@app.route("/api/detect", methods=["POST"])
def api_detect():

    if "image" not in request.files:
        return jsonify({"error": "no image provided"}), 400

    file = request.files["image"]

    npimg = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    results = model.predict(
        npimg,
        conf=0.25,
        verbose=False
    )

    fire = len(results[0].boxes) > 0

    confidence = 0.0

    if fire:
        confidence = float(results[0].boxes[0].conf[0])

    return jsonify({
        "fire": fire,
        "confidence": confidence
    })



# =========================
# 🚀 RUN
# =========================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False

    )
