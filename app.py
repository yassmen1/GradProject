from logging import config

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    session,
    send_file,
)
import cv2
import mediapipe as mp
import io
import json
import os
import requests
import time
import uuid
import pdfkit
from flask import make_response
from dotenv import load_dotenv
from deepface import DeepFace

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os
from openai import OpenAI
from tensorflow.keras.models import load_model
import numpy as np

emotion_model = load_model("emotion_model.h5", compile=False)

labels = ["anger", "fear", "joy", "neutral"]


load_dotenv()
Api_key = os.getenv("APiKey")
client = OpenAI(api_key=Api_key)
app = Flask(__name__)
app.secret_key = "secret123"
@app.context_processor
def inject_user():
    return dict(username=session.get("user"))

# ---------------- USERS ----------------
users = {}


def save_users():
    with open("data/users.json", "w") as f:
        json.dump(users, f)


def load_users():
    global users
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            users = json.load(f)


# ---------------- TELEGRAM ----------------
def format_message(code):
    messages = {
        "LOW_HR": "⚠️ Low Heart Rate",
        "HIGH_ACC": "⚠️ Repeated Movement",
        "LOUD_MIC": "⚠️ Loud Crying"
    }
    return messages.get(code, code)
def send_alert(msg, sensor_data=None):
    TOKEN = os.getenv("token")
    chat_id = os.getenv("chat_id")
    print("TOKEN:", TOKEN)
    print("CHAT ID:", chat_id)

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    try:
        if sensor_data:
            full_msg = f"""
🚨 ALERT

{msg}

HR: {sensor_data.get('hr')}
ACC: {sensor_data.get('acc')}
MIC: {sensor_data.get('mic')}
"""
        else:
            full_msg = msg

        response = requests.post(url, data={"chat_id": chat_id, "text": full_msg})

        print("TELEGRAM STATUS:", response.text)

    except Exception as e:
        print("ERROR:", e)


# ---------------- SENSOR ----------------
sensor_data = {"hr": 0, "acc": 0, "mic": 0, "freq": 0, "alerts": [], "state": "Normal"}
sensor_history = []

# 🔥 counters (عدد التكرار)
low_hr_counter = 0
high_acc_counter = 0
high_mic_counter = 0

# 🔥 limits (امتى يبعت)
LOW_HR_LIMIT = 10
HIGH_ACC_LIMIT = 5
HIGH_MIC_LIMIT = 5

# 🔥 cooldown لكل alert
last_alert_time = {}
ALERT_COOLDOWN = 10

def analyze_sensor(hr, acc, mic):
    global low_hr_counter, high_acc_counter, high_mic_counter

    alerts = []

    # ❤️ LOW HR
    if hr != 0 and hr < 60:
        low_hr_counter += 1
    else:
        low_hr_counter = 0

    if low_hr_counter >= LOW_HR_LIMIT:
        alerts.append("LOW_HR")
        low_hr_counter = 0

    # 🏃 MOVEMENT
    if acc > 15:
        high_acc_counter += 1
    else:
        high_acc_counter = 0

    if high_acc_counter >= HIGH_ACC_LIMIT:
        alerts.append("HIGH_ACC")
        high_acc_counter = 0

    # 🔊 MIC
    if mic > 70:
        high_mic_counter += 1
    else:
        high_mic_counter = 0

    if high_mic_counter >= HIGH_MIC_LIMIT:
        alerts.append("LOUD_MIC")
        high_mic_counter = 0

    return alerts


def predict_future():
    if len(sensor_history) < 5:
        return "Not enough data"

    hr_vals = [x["hr"] for x in sensor_history]
    acc_vals = [x["acc"] for x in sensor_history]
    mic_vals = [x["mic"] for x in sensor_history]

    hr_trend = hr_vals[-1] - hr_vals[0]
    acc_trend = acc_vals[-1] - acc_vals[0]
    mic_trend = mic_vals[-1] - mic_vals[0]

    if hr_trend > 20 and mic_trend > 15:
        return "Possible Meltdown Soon"

    if acc_trend > 10:
        return "Child becoming restless"

    return "Stable"


def predict_state(hr, acc, mic):
    if hr > 130 and mic > 60:
        return "Stress"
    if acc > 15 and mic > 70:
        return "Meltdown"
    return "Normal"


def emergency_mode(alerts):
    if len(alerts) >= 2:
        send_alert("🚨 EMERGENCY الحالة خطيرة!")


def log_data(data):
    with open("data/sensor_log.json", "a") as f:
        f.write(json.dumps(data) + "\n")


# ---------------- SENSOR API ----------------
@app.route("/sensor", methods=["POST"])
def sensor():
    global sensor_data, sensor_history

    data = request.json

    hr = data.get("hr", 0)
    acc = data.get("acc", 0)
    mic = data.get("mic", 0)
    freq = data.get("freq", 0)

    # خزني آخر readings
    sensor_history.append({"hr": hr, "acc": acc, "mic": mic})

    # خدي آخر 10 بس
    if len(sensor_history) > 10:
        sensor_history.pop(0)

    alerts = analyze_sensor(hr, acc, mic)
    state = predict_state(hr, acc, mic)

    # 🧠 prediction جديد
    prediction = predict_future()

    sensor_data = {
        "hr": hr,
        "acc": acc,
        "mic": mic,
        "freq": freq,
        "alerts": alerts,
        "state": state,
        "prediction": prediction,
    }
    log_data(sensor_data)
    emergency_mode(alerts)

    current_time = time.time()

    for a in alerts:

     if a not in last_alert_time:
        last_alert_time[a] = 0

     if current_time - last_alert_time[a] > ALERT_COOLDOWN:
        send_alert(format_message(a), sensor_data)
        last_alert_time[a] = current_time
    
    return {"status": "ok"}


@app.route("/get_data")
def get_data():
    return jsonify(sensor_data)


# ---------------- QUESTIONS ----------------
questions = [
    "Does your child respond when you call their name?",
    "Does your child imitate actions (like clapping or waving)?",
    "Does your child show appropriate emotions (smile, laugh)?",
    "Does your child show unusual body movements (rocking, spinning)?",
    "Does your child use toys in a normal way?",
    "Does your child get upset with small changes in routine?",
    "Does your child insist on doing things the same way every time?",
    "Does your child respond to sounds or voices?",
    "Does your child repeat words or phrases (echolalia)?",
    "Does your child show unusual fears or anxiety?",
    "Does your child speak or try to communicate verbally?",
    "Does your child use eye contact or gestures to communicate?",
    "Is your child overly active or unusually inactive?",
    "Are your child's behaviors consistent day to day?",
    "Overall, does your child behave like other children of same age?",
]

negative_questions = [3, 5, 8, 9, 12]

mapping = [
    "Relating",
    "Imitation",
    "Emotion",
    "Body",
    "Objects",
    "Adaptation",
    "Visual",
    "Listening",
    "Sensory",
    "Fear",
    "Verbal",
    "NonVerbal",
    "Activity",
    "Consistency",
    "General",
]


def eye_to_score(p):
    if p >= 70:
        return 1
    elif p >= 50:
        return 2
    elif p >= 30:
        return 3
    else:
        return 4


def adjust_score(i, val):
    if i in negative_questions:
        return 5 - val
    return val


def calculate_categories(answers, eye_score):
    categories = {k: [] for k in mapping}
    categories["Visual"] = eye_score

    for i, cat in enumerate(mapping):
        if cat != "Visual":
            categories[cat].append(adjust_score(i, answers[i]))

    for k in categories:
        if isinstance(categories[k], list):
            categories[k] = (
                sum(categories[k]) / len(categories[k]) if categories[k] else 1
            )

    return categories


def total_score(c):
    return sum(c.values())


def final_diagnosis(categories, eye_percent, emotion):

    autism_score = 0
    adhd_score = 0
    delay_score = 0

    # AUTISM
    if categories["Relating"] > 2.5:
        autism_score += 2
    if categories["Verbal"] > 2.5:
        autism_score += 2
    if categories["NonVerbal"] > 2.5:
        autism_score += 2
    if eye_percent < 40:
        autism_score += 2

    # 🔥 behavior
    if emotion in ["fear", "anger"]:
        autism_score += 1

    # ADHD
    if categories["Activity"] > 3:
        adhd_score += 2

    if emotion == "anger":
        adhd_score += 1

    # DELAY
    avg_score = sum(categories.values()) / len(categories)
    if avg_score > 2.5:
        delay_score += 3

    if emotion == "neutral":
        delay_score += 1

    # FINAL
    if autism_score >= 6:
        return "Autism"

    if adhd_score >= 4:
        return "ADHD"

    if delay_score >= 3:
        return "Delay"

    return "Normal"

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("dashboard.html", data=sensor_data)


# ---------------- MEDIAPIPE ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


# ---------------- EYE TRACKING ----------------
def run_eye_tracker():
    cap = cv2.VideoCapture(0)

    contact = 0
    total = 0

    emotion_counts = {
        "anger": 0,
        "fear": 0,
        "joy": 0,
        "neutral": 0
    }

    start_time = time.time()
    duration = 30
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        gaze_center = False

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            def xy(lm):
                return int(lm.x * w), int(lm.y * h)

            # -------- EMOTION --------
            try:
                x_coords = [int(lm.x * w) for lm in face.landmark]
                y_coords = [int(lm.y * h) for lm in face.landmark]

                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)

                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size != 0:

                    # 🔥 zoom
                    face_crop = cv2.resize(face_crop, None, fx=2.0, fy=2.0)

                    frame_count += 1

                    # حلل كل 3 frames
                    if frame_count % 3 == 0:

                        result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                        emotions = result[0]['emotion']

                        # -------- SMART LOGIC --------
                        happy = emotions.get("happy", 0)
                        angry = emotions.get("angry", 0)
                        fear = emotions.get("fear", 0)
                        sad = emotions.get("sad", 0)
                        surprise = emotions.get("surprise", 0)
                        neutral = emotions.get("neutral", 0)

                        # 🔥 تحديد الإيموشن
                        if angry > 10:
                            final = "anger"

                        elif happy > 15:
                            final = "joy"

                        elif fear > 20:
                            final = "fear"

                        elif sad > 25:
                            final = "neutral"

                        elif surprise > 20:
                            final = "joy"

                        else:
                            final = "neutral"

                        emotion_counts[final] += 1

            except:
                pass

            # -------- EYE CONTACT --------
            try:
                lpx, _ = xy(face.landmark[468])
                rpx, _ = xy(face.landmark[473])

                lx1, _ = xy(face.landmark[33])
                lx2, _ = xy(face.landmark[133])

                rx1, _ = xy(face.landmark[362])
                rx2, _ = xy(face.landmark[263])

                lw = lx2 - lx1
                rw = rx2 - rx1

                if lw > 0 and rw > 0:
                    left_ratio = (lpx - lx1) / lw
                    right_ratio = (rpx - rx1) / rw

                    if 0.35 < left_ratio < 0.65 and 0.35 < right_ratio < 0.65:
                        gaze_center = True

            except:
                gaze_center = False

        if gaze_center:
            contact += 1

        total += 1

        cv2.imshow("Eye Tracking", frame)

        if time.time() - start_time >= duration:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    percent = (contact / total) * 100 if total else 0

    # -------- FINAL EMOTION --------
    if sum(emotion_counts.values()) == 0:
        final_emotion = "neutral"
    else:
        final_emotion = max(emotion_counts, key=emotion_counts.get)

    print("Emotion Counts:", emotion_counts)
    print("Final Emotion:", final_emotion)

    return round(percent, 2), final_emotion

# ---------------- AUTH ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        u = request.form.get("username")
        p = request.form.get("password")

        if u in users:
            return "User exists"

        users[u] = {
            "password": p,
            "answers": [],
            "last_result": None,
            "chat_history": [],
        }

        save_users()
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username")
        p = request.form.get("password")

        if u in users and users[u]["password"] == p:
            session["user"] = u
            return redirect(url_for("index"))

        return "Wrong credentials"

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


def get_current_user():
    if "user" not in session:
        return None
    return session["user"]


# ---------------- MAIN ----------------
@app.route("/")
def index():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    user_data = users.get(user, {})
    result = user_data.get("last_result", "No result yet")  # 🔥 دي أهم سطر

    return render_template("index.html", result=result)


@app.route("/parent_info", methods=["GET", "POST"])
def parent_info():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    # لو المستخدم عمل assessment قبل كده
    if users[user]["last_result"] is not None:
        return redirect(url_for("result"))

    if request.method == "POST":
        users[user]["answers"] = [None] * len(questions)
        save_users()
        return redirect(url_for("question", q=0))

    return render_template("parent_info.html")


@app.route("/questionnaire", methods=["GET", "POST"])
def question():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))
    if users[user]["last_result"] is not None:
        return redirect(url_for("specialist"))

    q = int(request.args.get("q", 0))

    if request.method == "POST":

        if "prev" in request.form:
            return redirect(url_for("question", q=max(0, q - 1)))

        if "next" in request.form:
            ans = request.form.get("answer")

            if not ans:
                return render_template(
                    "question.html",
                    question=questions[q],
                    q_index=q,
                    total=len(questions),
                    selected=users[user]["answers"][q],
                    error="Please select an answer first",
                )

            users[user]["answers"][q] = int(ans)
            save_users()

            if q == len(questions) - 1:
                return redirect(url_for("eye_test"))

            return redirect(url_for("question", q=q + 1))

    return render_template(
        "question.html",
        question=questions[q],
        q_index=q,
        total=len(questions),
        selected=users[user]["answers"][q],
        error=None,
        username=user
    )


@app.route("/eye_test", methods=["GET", "POST"])
def eye_test():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    if users[user]["last_result"] is not None:
        return redirect(url_for("specialist"))

    if request.method == "POST":

        eye, emotion = run_eye_tracker()

        score_eye = eye_to_score(eye)

        categories = calculate_categories(users[user]["answers"], score_eye)

        diagnosis = final_diagnosis(categories, eye, emotion)

        cars = total_score(categories)
        percent = (cars / 60) * 100

        users[user]["last_result"] = {
            "diagnosis": diagnosis,
            "percentage": percent,
            "eye_percent": eye,
            "categories": categories,
            "emotion": emotion,
        }

        save_users()

        return redirect(url_for("result"))

    # ✅ ده أهم سطر (كان ناقص)
    return render_template("eye_test.html")

    


# 🔥 CHAT
@app.route("/ask_ai", methods=["POST"])
def ask_ai():
    user = get_current_user()
    if not user:
        return jsonify({"answer": "Please login again"})

    question = request.json.get("question")
    data = users[user]["last_result"]

    context = f"""
Diagnosis: {data['diagnosis']}
Severity: {data['percentage']}%
Eye Contact: {data['eye_percent']}%

HR: {sensor_data['hr']}
Movement: {sensor_data['acc']}
Sound: {sensor_data['mic']}
State: {sensor_data['state']}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an autism specialist helping parents.",
                },
                {"role": "user", "content": context + "\n\n" + question},
            ],
        )

        answer = response.choices[0].message.content

    except Exception as e:
        print("ERROR:", e)
        answer = "AI not working"

    users[user]["chat_history"].append({"role": "user", "content": question})
    users[user]["chat_history"].append({"role": "assistant", "content": answer})

    save_users()

    return jsonify({"answer": answer})


@app.route("/voice_ai", methods=["POST"])
def voice_ai():

    user = get_current_user()
    if not user:
        return jsonify({"answer": "Login first"})

    audio = request.files["audio"]

    import uuid
    import os

    os.makedirs("static/audio", exist_ok=True)

    filename = f"static/audio/{uuid.uuid4()}.webm"
    audio.save(filename)

    try:
        transcript = client.audio.transcriptions.create(
            file=open(filename, "rb"), model="gpt-4o-mini-transcribe"
        )

        text = transcript.text

        data = users[user]["last_result"]

        context = f"""
Diagnosis: {data['diagnosis']}
Severity: {data['percentage']}%
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are autism specialist."},
                {"role": "user", "content": context + "\n\n" + text},
            ],
        )

        answer = response.choices[0].message.content

    except Exception as e:
        print("ERROR:", e)
        answer = "Voice failed"

    # 🔥 حفظ الصوت + الرد
    if "chat_history" not in users[user]:
        users[user]["chat_history"] = []

    users[user]["chat_history"].append({"type": "audio", "url": "/" + filename})

    users[user]["chat_history"].append({"type": "ai", "text": answer})

    return jsonify({"answer": answer, "audio_url": "/" + filename})


@app.route("/specialist")
def specialist():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    data = users[user]["last_result"]
    chat = users[user]["chat_history"]

    if not data:
        return redirect(url_for("index"))

    return render_template("specialist.html", **data, chat_history=chat)


# 🔥 PDF
@app.route("/download_pdf")
def download_pdf():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    data = users[user]["last_result"]

    rendered = render_template(
        "result_pdf.html",
        diagnosis=data["diagnosis"],
        percentage=round(data["percentage"], 1),
        eye_percent=data["eye_percent"],
        categories=data["categories"],
    )
    config = pdfkit.configuration(
        wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
    )

    pdf = pdfkit.from_string(rendered, False, configuration=config)

    response = make_response(pdf)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=report.pdf"

    return response
@app.route("/result")
def result():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    data = users[user]["last_result"]

    return render_template(
        "result.html",
        diagnosis=data["diagnosis"],
        percentage=round(data["percentage"], 1),
        eye_percent=data["eye_percent"],
        categories=data["categories"],
        emotion=data["emotion"]
    )

# ---------------- START ----------------
load_users()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)