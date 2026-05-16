from datetime import datetime
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
import re
from flask import make_response
from dotenv import load_dotenv
from deepface import DeepFace

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.lib.styles import getSampleStyleSheet
import os
from openai import OpenAI
from tensorflow.keras.models import load_model
import numpy as np

emotion_model = load_model("emotion_model.h5", compile=False)

labels = ["anger", "fear", "joy", "neutral"]


load_dotenv()
os.makedirs("data", exist_ok=True)
Api_key = os.getenv("APiKey")
client = OpenAI(api_key=Api_key)
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")


@app.context_processor
def inject_user():
    return dict(
        username=session.get("user"),
        get_text=get_text
    )


# ---------------- USERS ----------------
users = {}


def save_users():
    with open("data/users.json", "w") as f:
        json.dump(users, f)


def load_users():
    global users
    if os.path.exists("data/users.json"):
        with open("data/users.json", "r") as f:
            users = json.load(f)


# ---------------- TELEGRAM ----------------
def format_message(code):
    messages = {
        "LOW_HR": get_text("⚠️ انخفاض في ضربات القلب", "⚠️ Low Heart Rate"),
        "HIGH_ACC": get_text("⚠️ حركة زائدة متكررة", "⚠️ Repeated Movement"),
        "LOUD_MIC": get_text("⚠️ صوت مرتفع (بكاء)", "⚠️ Loud Crying"),
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
ALERT_COOLDOWN = 20
last_sent_state = ""
active_alerts = set()

def analyze_sensor(hr, acc, mic):

    global low_hr_counter, high_acc_counter, high_mic_counter
    # 📈 HR TREND ANALYSIS
    recent_hr = [
    x["hr"]
    for x in sensor_history[-8:]
    if x["hr"] > 50 and x["hr"] < 180
]
    hr_rising_fast = False

    if len(recent_hr) >= 5:

        # متوسط أول قراءتين
        avg_old = sum(recent_hr[:2]) / 2

        # متوسط آخر قراءتين
        avg_new = sum(recent_hr[-2:]) / 2

        hr_change = avg_new - avg_old
        print("HR CHANGE:", hr_change)
        print("RECENT HR:", recent_hr)

        # 🔥 لو HR زاد فجأة
        if hr_change > 10:
            hr_rising_fast = True
    alerts = []

    # ❤️ LOW HR
    if hr != 0 and hr < 60:
        low_hr_counter += 1
    else:
        low_hr_counter = 0

    if low_hr_counter >= LOW_HR_LIMIT:
        alerts.append("LOW_HR")
        low_hr_counter = 0

    # 🚨 SMART MELTDOWN
    if hr_rising_fast and acc > 13 and mic > 5:
        alerts.append("MELTDOWN")

    # 🏃 MOVEMENT
    if acc > 15:
        high_acc_counter += 1
    else:
        high_acc_counter = 0

    if high_acc_counter >= HIGH_ACC_LIMIT:
        alerts.append("HIGH_ACC")
        high_acc_counter = 0

    # 🔊 MIC
    if mic > 8:
        high_mic_counter += 1
    else:
        high_mic_counter = 0

    if high_mic_counter >= HIGH_MIC_LIMIT:
        alerts.append("LOUD_MIC")
        high_mic_counter = 0

    
    # 🧠 SMART STRESS DETECTION
    if hr_rising_fast and mic > 5:
        alerts.append("STRESS")

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

    # 🔴 Meltdown
    if acc > 15 and mic > 8 and hr > 100:
        return "Meltdown"

    # 🟠 Stress
    if hr > 95 and mic >= 5:
        return "Stress"

    # 🟡 Active
    if acc >= 11 and acc < 14 and mic < 5:
        return "Active"

    # 🟢 Calm
    if hr >= 60 and hr <= 95 and acc < 11 and mic < 3:
        return "Calm"

    return "Normal"


def emergency_mode(alerts):
    if len(alerts) >= 2:
        send_alert(get_text("🚨 حالة طارئة!", "🚨 EMERGENCY!"))


def log_data(data):
    with open("data/sensor_log.json", "a") as f:
        f.write(json.dumps(data) + "\n")


# ---------------- SENSOR API ----------------
@app.route("/sensor", methods=["POST"])
def sensor():
    global sensor_data, sensor_history, active_alerts, last_sent_state

    data = request.json

    hr = data.get("hr", 0)
    acc = data.get("acc", 0)
    mic = data.get("mic", 0)
    freq = data.get("freq", 0)

    # خزني آخر readings
    sensor_history.append({"hr": hr, "acc": acc, "mic": mic})

    # خدي آخر 10 بس
    if len(sensor_history) > 50:
        sensor_history.pop(0)

    alerts = analyze_sensor(hr, acc, mic)
    # 🔥 تحديد الحالة بناءً على الـ alerts

    state = "Normal"

    if "MELTDOWN" in alerts:
        state = "Meltdown"

    elif "STRESS" in alerts:
        state = "Stress"

    elif "HIGH_ACC" in alerts:
        state = "Active"

    elif hr >= 60 and hr <= 95 and mic < 3:
        state = "Calm"

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
    #emergency_mode(alerts)

    current_time = time.time()

    # أول مرة
    if state not in last_alert_time:
        last_alert_time[state] = 0

    # الحالات المهمة فقط
    if state not in ["Normal", "Calm"]:

        # لو عدى 30 ثانية
        if current_time - last_alert_time[state] > 30:

            alert_message = f"🚨 {state}"

            send_alert(alert_message, sensor_data)

            last_alert_time[state] = current_time

    return {"status": "ok"}


@app.route("/get_data")
def get_data():
    return jsonify(sensor_data)


@app.route("/sensor_history")
def history():
    return jsonify(sensor_history)


# ---------------- QUESTIONS ----------------
questions_en = [
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

questions_ar = [
    "هل يستجيب طفلك عند مناداته باسمه؟",
    "هل يقلد طفلك الأفعال (مثل التصفيق أو التلويح)؟",
    "هل يظهر طفلك مشاعر مناسبة (ابتسامة، ضحك)؟",
    "هل يقوم طفلك بحركات جسدية غير معتادة (مثل التأرجح أو الدوران)؟",
    "هل يستخدم طفلك الألعاب بطريقة طبيعية؟",
    "هل ينزعج طفلك من التغييرات البسيطة في الروتين؟",
    "هل يصر طفلك على القيام بالأشياء بنفس الطريقة دائمًا؟",
    "هل يستجيب طفلك للأصوات أو النداء؟",
    "هل يكرر طفلك الكلمات أو الجمل (ترديد الكلام)؟",
    "هل يظهر طفلك مخاوف أو قلق غير معتاد؟",
    "هل يتحدث طفلك أو يحاول التواصل لفظيًا؟",
    "هل يستخدم طفلك التواصل البصري أو الإشارات للتواصل؟",
    "هل طفلك نشيط جدًا أو خامل بشكل غير طبيعي؟",
    "هل سلوك طفلك ثابت يومًا بعد يوم؟",
    "بشكل عام، هل يتصرف طفلك مثل الأطفال في نفس عمره؟",
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

    if emotion in ["anger", "joy", "neutral","fear"]:
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

    if adhd_score >= 3:
        return "ADHD"

    if delay_score >= 2:
        return "Delay"

    return "Normal"


# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    lang = session.get("lang", "en")
    return render_template(
        "dashboard.html",
        username=session["user"],
        lang=lang,
        # navbar
        welcome_text=get_text("مرحبًا", "Welcome"),
        logout_text=get_text("تسجيل خروج", "Logout"),
        # title
        title=get_text("لوحة التحكم", "Dashboard"),
        live_text=get_text("المتابعة المباشرة", "Live Monitoring Dashboard"),
        # cards
        hr_text=get_text("معدل ضربات القلب", "Heart Rate"),
        acc_text=get_text("التسارع", "Acceleration"),
        mic_text=get_text("الميكروفون", "Microphone"),
        freq_text=get_text("التردد", "Frequency"),
        # status
        state_text=get_text("الحالة الحالية", "Current Stat"),
        prediction_text=get_text("التوقع", "Prediction"),
        # alerts
        no_alerts=get_text("لا يوجد تنبيهات", "No alerts"),
    )


@app.route("/history")
def assessment_history():

    user = get_current_user()

    if not user:
        return redirect(url_for("login"))

    assessments = users[user].get("assessments", [])

    lang = session.get("lang", "en")

    diagnosis_map = {
        "Autism": get_text("توحد", "Autism"),
        "Delay": get_text("تأخر نمائي", "Delay"),
        "Normal": get_text("طبيعي", "Normal"),
        "Mild": get_text("بسيط", "Mild"),
        "Moderate": get_text("متوسط", "Moderate"),
        "Severe": get_text("شديد", "Severe"),
    }

    for a in assessments:

        diagnosis_value = a.get("diagnosis", "")

        a["diagnosis"] = diagnosis_map.get(
            diagnosis_value,
            diagnosis_value
        )

    return render_template(
        "history.html",
        assessments=assessments,
        username=user,
        lang=lang,
        welcome_text=get_text("مرحبًا", "Welcome"),
        logout_text=get_text("تسجيل خروج", "Logout"),
    )
@app.route("/continue")
def continue_options():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "continue.html",
        username=session["user"],
        lang=session.get("lang", "en"),
        welcome_text=get_text("مرحبًا", "Welcome"),
        logout_text=get_text("تسجيل خروج", "Logout"),
    )


# ---------------- MEDIAPIPE ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


# ---------------- EYE TRACKING ----------------
def run_eye_tracker():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available")

        return 0, "neutral"

    contact = 0
    total = 0

    emotion_counts = {"anger": 0, "fear": 0, "joy": 0, "neutral": 0}

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

                        result = DeepFace.analyze(
                            face_crop, actions=["emotion"], enforce_detection=False
                        )
                        emotions = result[0]["emotion"]

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

            except Exception as e:
                print("Emotion Error:", e)
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

            except Exception as e:
                print("Eye Tracking Error:", e)
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
# 🌍 language
@app.route("/set_lang/<lang>")
def set_lang(lang):
    session["lang"] = lang
    return redirect(request.referrer)


def get_text(ar, en):
    if session.get("lang") == "ar":
        return ar
    return en


@app.context_processor
def inject_lang():
    return dict(lang=session.get("lang", "en"))


@app.route("/register", methods=["GET", "POST"])
def register():

    error = None

    if request.method == "POST":

        u = request.form.get("username", "").strip()
        p = request.form.get("password", "").strip()

        # USERNAME VALIDATION
        if len(u) < 4:

            error = get_text(
                "اسم المستخدم لازم يكون 4 حروف على الأقل",
                "Username must be at least 4 characters",
            )

        elif " " in u:

            error = get_text(
                "اسم المستخدم لا يجب أن يحتوي على مسافات",
                "Username cannot contain spaces",
            )

        elif not re.match(r"^[A-Za-z0-9_]+$", u):

            error = get_text(
                "اسم المستخدم لازم يحتوي على حروف وأرقام و _ فقط",
                "Username can contain only letters, numbers and _",
            )

        # PASSWORD VALIDATION
        elif len(p) < 8:

            error = get_text(
                "الباسورد لازم يكون 8 حروف على الأقل",
                "Password must be at least 8 characters",
            )

        elif not re.search(r"[A-Z]", p):

            error = get_text(
                "الباسورد لازم يحتوي على حرف كبير",
                "Password must contain at least one uppercase letter",
            )

        elif not re.search(r"[a-z]", p):

            error = get_text(
                "الباسورد لازم يحتوي على حرف صغير",
                "Password must contain at least one lowercase letter",
            )

        elif not re.search(r"[0-9]", p):

            error = get_text(
                "الباسورد لازم يحتوي على رقم",
                "Password must contain at least one number",
            )

        elif not re.search(r"[!@#$%^&*(),.?\":{}|<>]", p):

            error = get_text(
                "الباسورد لازم يحتوي على رمز خاص",
                "Password must contain at least one special character",
            )

        # USER EXISTS
        elif u in users:

            error = get_text(
                "المستخدم موجود بالفعل",
                "User already exists",
            )

        # SUCCESS
        if not error:

            users[u] = {
                "password": generate_password_hash(p),
                "answers": [],
                "last_result": None,
                "chat_history": [],
                "assessments": [],
            }

            save_users()

            return redirect(url_for("login"))

    return render_template(
        "register.html",

        error=error,

        title=get_text("إنشاء حساب", "Create Account"),
        subtitle=get_text("سجّل لإنشاء حساب", "Register to continue"),
        btn_text=get_text("تسجيل", "Sign Up"),

        username_text=get_text("اسم المستخدم", "Username"),
        password_text=get_text("كلمة المرور", "Password"),

        have_account_text=get_text("عندك حساب؟", "Already have account?"),
        login_text=get_text("تسجيل الدخول", "Login"),
    )


@app.route("/login", methods=["GET", "POST"])
def login():

    error = None  # 🔥 مهم

    if request.method == "POST":
        u = request.form.get("username")
        p = request.form.get("password")

        if u in users:

            saved_password = users[u]["password"]

            # ✅ users الجديدة (hashed)
            if saved_password.startswith("pbkdf2:") or saved_password.startswith("scrypt:"):

                if check_password_hash(saved_password, p):

                    session["user"] = u

                    if users[u].get("assessments"):
                        return redirect("/continue")

                    return redirect(url_for("index"))

            # ✅ users القديمة (plain text)
            else:

                if saved_password == p:

                    # 🔥 نحولها تلقائي لـ hash
                    users[u]["password"] = generate_password_hash(p)
                    save_users()

                    session["user"] = u

                    if users[u].get("assessments"):
                        return redirect("/continue")

                    return redirect(url_for("index"))

        # ❌ لو غلط
        error = get_text("بيانات غير صحيحة", "Wrong credentials")

    return render_template(
        "login.html",
        # 🧠 left side
        title_main_line1=get_text("نظام ذكي", "Smart Autism"),
        title_main_line2=get_text("لمتابعة التوحد", "Monitoring System"),
        description=get_text(
            "نظام ذكي لمتابعة سلوك الطفل وتحليل حالته باستخدام الذكاء الاصطناعي",
            "AI-powered system to monitor behavior, detect stress, and help caregivers understand the child in real time.",
        ),
        # features
        f1_title=get_text("مراقبة لحظية", "Real-time monitoring"),
        f1_desc=get_text(
            "متابعة مباشرة وتنبيهات فورية", "Live tracking and instant alerts"
        ),
        f2_title=get_text("تحليلات ذكية", "Smart analytics"),
        f2_desc=get_text(
            "تقارير وتحليل باستخدام الذكاء الاصطناعي",
            "AI insights and advanced reports",
        ),
        f3_title=get_text("تشخيص ذكي", "AI diagnosis"),
        f3_desc=get_text(
            "تقييم وتحليل دقيق للحالة", "Intelligent assessment and predictions"
        ),
        # login card
        login_title=get_text("نظام التوحد", "Smart Autism System"),
        subtitle=get_text("سجل الدخول للمتابعة", "Login to continue"),
        username_text=get_text("اسم المستخدم", "Username"),
        password_text=get_text("كلمة المرور", "Password"),
        btn_text=get_text("تسجيل الدخول", "Login"),
        register_text=get_text("إنشاء حساب", "Create account"),
        error=error,  # 🔥 مهم
        lang=session.get("lang", "en"),
    )


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
    result = user_data.get("last_result", "No result yet")

    title = get_text("نظام التوحد", "Autism System")
    no_result = get_text("لا يوجد نتيجة", "No result yet")

    return render_template(
        "index.html",
        # القديم (سيبيه)
        result=result or no_result,
        title=title,
        # 👤 user
        username=user,
        # 🔝 navbar
        welcome_text=get_text("أهلاً", "Welcome"),
        logout_text=get_text("تسجيل الخروج", "Logout"),
        # 🧠 hero
        title_main_line1=get_text("نظام ذكي", "Smart Autism"),
        title_main_line2=get_text("لمتابعة التوحد", "Monitoring System"),
        description=get_text(
            "نظام ذكي لتقييم ومتابعة التوحد باستخدام الذكاء الاصطناعي",
            "An intelligent autism assessment system combining AI diagnosis, eye tracking, and real-time monitoring",
        ),
        start_text=get_text("ابدأ التقييم", "Start Assessment"),
    )

@app.route("/start_assessment")
def start_assessment():

    # لو مش عامل login
    if "user" not in session:
        return redirect(url_for("login"))

    user = session["user"]

    # لو عنده assessment قديم
    if user in users and users[user].get("assessments"):

        return redirect(url_for("continue_options"))

    # لو أول مرة
    return redirect(url_for("parent_info"))
@app.route("/parent_info", methods=["GET", "POST"])
def parent_info():

    user = get_current_user()

    if not user:
        return redirect(url_for("login"))

    if request.method == "POST":

        age = int(request.form.get("age"))
        gender = request.form.get("gender")

        if age < 2 or age > 12:

            return render_template(
                "parent_info.html",

                error=get_text(
                    "العمر يجب أن يكون بين 2 و 12",
                    "Age must be between 2 and 12"
                ),

                title=get_text("نظام التوحد", "Autism System"),
                welcome_text=get_text("أهلاً", "Welcome"),
                logout_text=get_text("تسجيل الخروج", "Logout"),
                username=user,

                title_info=get_text("بيانات الطفل", "Child Information"),
                age_text=get_text("العمر", "Age"),
                gender_text=get_text("اختر النوع", "Select Gender"),
                male_text=get_text("ذكر", "Male"),
                female_text=get_text("أنثى", "Female"),
                start_text=get_text("ابدأ الأسئلة", "Start Questions")
            )

        if not gender:

            return render_template(
                "parent_info.html",

                error=get_text(
                    "اختر النوع",
                    "Please select gender"
                ),

                title=get_text("نظام التوحد", "Autism System"),
                welcome_text=get_text("أهلاً", "Welcome"),
                logout_text=get_text("تسجيل الخروج", "Logout"),
                username=user,

                title_info=get_text("بيانات الطفل", "Child Information"),
                age_text=get_text("العمر", "Age"),
                gender_text=get_text("اختر النوع", "Select Gender"),
                male_text=get_text("ذكر", "Male"),
                female_text=get_text("أنثى", "Female"),
                start_text=get_text("ابدأ الأسئلة", "Start Questions")
            )

        users[user]["age"] = age
        users[user]["gender"] = gender
        users[user]["answers"] = [None] * len(questions_en)

        save_users()

        return redirect(url_for("question", q=0))

    return render_template(
        "parent_info.html",

        title=get_text("نظام التوحد", "Autism System"),
        welcome_text=get_text("أهلاً", "Welcome"),
        logout_text=get_text("تسجيل الخروج", "Logout"),
        username=user,

        title_info=get_text("بيانات الطفل", "Child Information"),
        age_text=get_text("العمر", "Age"),
        gender_text=get_text("اختر النوع", "Select Gender"),
        male_text=get_text("ذكر", "Male"),
        female_text=get_text("أنثى", "Female"),
        start_text=get_text("ابدأ الأسئلة", "Start Questions"),
    )
@app.route("/questionnaire", methods=["GET", "POST"])
def question():

    user = get_current_user()
    if not user:
        return redirect(url_for("login"))
    # if users[user]["last_result"] is not None:
    # return redirect(url_for("specialist"))

    q = int(request.args.get("q", 0))

    # 🔥 texts
    questions_list = questions_ar if session.get("lang") == "ar" else questions_en

    next_text = get_text("التالي", "Next")
    prev_text = get_text("السابق", "Previous")
    error_text = get_text("اختار إجابة الأول", "Please select an answer first")
    question_title = get_text("السؤال", "Question")
    of_text = get_text("من", "of")

    opt1 = get_text("دائمًا", "Always")
    opt2 = get_text("أحيانًا", "Sometimes")
    opt3 = get_text("نادرًا", "Rarely")
    opt4 = get_text("أبدًا", "Never")

    finish_text = get_text("إنهاء", "Finish")

    if request.method == "POST":

        if "prev" in request.form:
            return redirect(url_for("question", q=max(0, q - 1)))

        if "next" in request.form:
            ans = request.form.get("answer")

            if not ans:
                return render_template(
                    "question.html",
                    question=questions_list[q],
                    q_index=q,
                    total=len(questions_en),
                    selected=users[user]["answers"][q],
                    error=error_text,
                    next_text=next_text,
                    prev_text=prev_text,
                    username=user,
                    question_title=question_title,
                    of_text=of_text,
                    opt1=opt1,
                    opt2=opt2,
                    opt3=opt3,
                    opt4=opt4,
                    finish_text=finish_text,
                )

            users[user]["answers"][q] = int(ans)
            save_users()

            if q == len(questions_en) - 1:
                return redirect(url_for("eye_test"))

            return redirect(url_for("question", q=q + 1))

    return render_template(
        "question.html",
        question=questions_list[q],
        q_index=q,
        total=len(questions_en),
        selected=users[user]["answers"][q],
        error=None,
        username=user,
        next_text=next_text,
        prev_text=prev_text,
        question_title=question_title,
        of_text=of_text,
        opt1=opt1,
        opt2=opt2,
        opt3=opt3,
        opt4=opt4,
        finish_text=finish_text,
    )


@app.route("/eye_test", methods=["GET", "POST"])
def eye_test():

    user = get_current_user()

    if not user:
        return redirect(url_for("login"))

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

        # 🔥 assessment object
        new_assessment = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "diagnosis": diagnosis,
            "percentage": percent,
            "eye_percent": eye,
            "categories": categories,
            "emotion": emotion,
        }

        # 🔥 AI comparison
        analysis_text = ""

        old_assessments = users[user].get("assessments", [])

        if len(old_assessments) > 0:

            prev = old_assessments[-1]

            prompt = f"""
Previous assessment:
Autism indicators: {prev['percentage']}%
Eye contact: {prev['eye_percent']}%
Emotion: {prev['emotion']}

Current assessment:
Autism indicators: {percent}%
Eye contact: {eye}%
Emotion: {emotion}

Analyze the child's progress briefly like an autism specialist.

Mention:
- improvement or decline
- eye contact changes
- emotional changes
- recommendation

Keep the response short and professional.
Respond in Arabic if the system language is Arabic.
"""

            try:

                ai_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an autism specialist.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                )

                analysis_text = ai_response.choices[0].message.content

            except Exception as e:

                print("AI ERROR:", e)

                analysis_text = "AI analysis unavailable."

        # 🔥 save AI analysis
        new_assessment["ai_analysis"] = analysis_text

        # 🔥 save assessment
        users[user].setdefault("assessments", [])

        users[user]["assessments"].append(new_assessment)

        save_users()

        return redirect(url_for("result"))

    lang = session.get("lang", "en")

    return render_template(
        "eye_test.html",
        username=user,
        lang=lang,
        title=get_text("اختبار التواصل البصري 👁️", "Eye Contact Test 👁️"),
        instruction=get_text(
            "انظر مباشرة إلى الكاميرا لبضع ثواني",
            "Please look directly at the camera for a few seconds.",
        ),
        btn_text=get_text("ابدأ الاختبار", "Start Test"),
        welcome_text=get_text("مرحبًا", "Welcome"),
        logout_text=get_text("تسجيل خروج", "Logout"),
    )


# 🔥 CHAT
@app.route("/ask_ai", methods=["POST"])
def ask_ai():
    user = get_current_user()
    if not user:
        return jsonify({"answer": get_text("سجّل دخول تاني", "Please login again")})

    question = request.json.get("question")
    if not question or not question.strip():
     return jsonify({"answer": "Empty message"})
    data = users[user].get("last_result")
    if not data:
        return jsonify(
            {
                "answer": get_text(
                    "اعمل التقييم الأول", "Please complete assessment first"
                )
            }
        )

    context = f"""
Diagnosis: {data['diagnosis']}
Severity: {data['percentage']}%
Eye Contact: {data['eye_percent']}%

HR: {sensor_data['hr']}
Movement: {sensor_data['acc']}
Sound: {sensor_data['mic']}
State: {sensor_data['state']}
"""
    system_msg = get_text(
        "أنت متخصص في التوحد تساعد أولياء الأمور",
        "You are an autism specialist helping parents",
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_msg,
                },
                {"role": "user", "content": context + "\n\n" + question},
            ],
        )

        answer = response.choices[0].message.content

    except Exception as e:
        print("ERROR:", e)
        answer = "AI not working"
        
    if "chat_history" not in users[user]:
        users[user]["chat_history"] = []
        
    users[user]["chat_history"].append({"role": "user", "content": question})
    users[user]["chat_history"].append({"role": "assistant", "content": answer})

    save_users()

    return jsonify({"answer": answer})


@app.route("/voice_ai", methods=["POST"])
def voice_ai():

    user = get_current_user()
    if not user:
        return jsonify({"answer": "Login first"})

    audio = request.files.get("audio")

    if not audio:
        return jsonify({"answer": "No audio uploaded"})

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

        data = users[user].get("last_result")
        if not data:
            return jsonify(
                {
                    "answer": get_text(
                        "اعمل التقييم الأول", "Please complete assessment first"
                    )
                }
            )

        context = f"""
Diagnosis: {data['diagnosis']}
Severity: {data['percentage']}%
"""
        system_msg1 = get_text("أنت متخصص في التوحد", "You are autism specialist")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg1},
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
    save_users()

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
    lang = session.get("lang", "en")
    
    diagnosis_map = {
    "Autism": get_text("توحد", "Autism"),
    "Delay": get_text("تأخر نمائي", "Delay"),
    "Normal": get_text("طبيعي", "Normal"),
    "Mild": get_text("بسيط", "Mild"),
    "Moderate": get_text("متوسط", "Moderate"),
    "Severe": get_text("شديد", "Severe"),
    }

    diagnosis = diagnosis_map.get(
        data["diagnosis"],
        data["diagnosis"]
    )
    return render_template(
    "specialist.html",
    username=user,
    lang=lang,
    chat_history=chat,

    percentage=data["percentage"],
    eye_percent=data["eye_percent"],
    emotion=data["emotion"],
    categories=data["categories"],

    autism_text=get_text("توحد", "Autism"),
    title=get_text("المتخصص", "Specialist"),
    send_text=get_text("إرسال", "Send"),
    welcome_text=get_text("مرحبًا", "Welcome"),
    logout_text=get_text("تسجيل خروج", "Logout"),
    severity_text=get_text("شدة الحالة", "Severity"),
    eye_text=get_text("التواصل البصري", "Eye Contact"),
    placeholder_text=get_text("اكتب رسالتك...", "Type your message..."),
    typing_text=get_text("يكتب...", "Typing..."),

    diagnosis=diagnosis,
)


# 🔥 PDF
@app.route("/download_pdf")
def download_pdf():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    data = users[user]["last_result"]
    if not data:    
        return redirect(url_for("index"))
    title = get_text("التقرير", "Report")
    diagnosis_title = get_text("التشخيص", "Diagnosis")
    eye_title = get_text("التواصل البصري", "Eye Contact")

    rendered = render_template(
        "result_pdf.html",
        diagnosis=data["diagnosis"],
        percentage=round(data["percentage"], 1),
        eye_percent=data["eye_percent"],
        categories=data["categories"],
        # 🔥 texts
        title=title,
        diagnosis_title=diagnosis_title,
        eye_title=eye_title,
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
    if not data:
        return redirect(url_for("index"))
    lang = session.get("lang", "en")  # 🔥 مهم
    diagnosis_map = {
        "Autism": get_text("توحد", "Autism"),
        "Normal": get_text("طبيعي", "Normal"),
        "Mild": get_text("بسيط", "Mild"),
        "Moderate": get_text("متوسط", "Moderate"),
        "Severe": get_text("شديد", "Severe"),
        "Severe": get_text("شديد", "Severe"),"Delay": get_text(" تأخر في النمو", "Delay"),
        
    }
    category_names = {
        "Relating": get_text("التفاعل", "Relating"),
        "Imitation": get_text("التقليد", "Imitation"),
        "Emotion": get_text("العاطفة", "Emotion"),
        "Body": get_text("الجسم", "Body"),
        "Objects": get_text("الأشياء", "Objects"),
        "Adaptation": get_text("التكيف", "Adaptation"),
        "Visual": get_text("البصري", "Visual"),
        "Listening": get_text("الاستماع", "Listening"),
        "Sensory": get_text("الحسي", "Sensory"),
        "Fear": get_text("الخوف", "Fear"),
        "Verbal": get_text("اللفظي", "Verbal"),
        "NonVerbal": get_text("غير اللفظي", "Non-Verbal"),
        "Activity": get_text("النشاط", "Activity"),
        "Consistency": get_text("الاستمرارية", "Consistency"),
        "General": get_text("عام", "General"),
    }
    return render_template(
        "result.html",
        username=user,
        lang=lang,
        # data
        diagnosis=diagnosis_map.get(data["diagnosis"], data["diagnosis"]),
        percentage=round(data["percentage"], 1),
        eye_percent=data["eye_percent"],
        categories=data["categories"],
        emotion=data["emotion"],
        # navbar
        welcome_text=get_text("مرحبًا", "Welcome"),
        logout_text=get_text("تسجيل خروج", "Logout"),
        # titles
        diagnosis_title=get_text("التشخيص", "Diagnosis"),
        autism_text=get_text("مؤشرات التوحد", "Autism Indicators"),
        eye_text=get_text("التواصل البصري", "Eye Contact"),
        emotion_text=get_text("الحالة العاطفية", "Emotion"),
        cars_text=get_text("درجة CARS", "CARS Score"),
        details_text=get_text("النتائج التفصيلية", "Detailed Scores"),
        # buttons
        home_text=get_text("الرئيسية", "Home"),
        specialist_text=get_text("الأخصائي", "Specialist"),
        pdf_text=get_text("تحميل PDF", "PDF"),
        monitor_text=get_text("المتابعة", "Monitor"),
        # message
        monitor_msg=get_text(
            "نوصي بالمتابعة المستمرة باستخدام السوار الذكي لفهم سلوك الطفل بشكل أفضل",
            "We recommend continuous monitoring using the smart band to better understand your child's behavior.",
        ),
        # High good calm
        status_high=get_text("مرتفع", "High"),
        status_good=get_text("جيد", "Good"),
        status_moderate=get_text("متوسط", "Moderate"),
        status_weak=get_text("ضعيف", "Weak"),
        status_very_low=get_text("ضعيف جدًا", "Very Low"),
        status_severe=get_text("شديد", "Severe"),
        status_calm=get_text("هادئ", "Calm"),
        status_positive=get_text("إيجابي", "Positive"),
        status_distress=get_text("توتر", "Distress"),
        status_anxiety=get_text("قلق", "Anxiety"),
        # 🔥 الجديد
        scores_meaning=get_text("ماذا تعني النتائج؟", "What do the scores mean?"),
        normal_text=get_text("طبيعي", "Normal"),
        mild_text=get_text("بسيط", "Mild"),
        moderate_text=get_text("متوسط", "Moderate"),
        severe_text=get_text("شديد", "Severe"),
        cars_interpret=get_text("تفسير CARS", "CARS Interpretation"),
        eye_interpret=get_text("تفسير التواصل البصري", "Eye Contact Interpretation"),
        autism_interpret=get_text(
            "تفسير مؤشرات التوحد", "Autism Indicators Interpretation"
        ),
        area_text=get_text("المجال", "Area"),
        score_text=get_text("الدرجة", "Score"),
        percent_text=get_text("النسبة", "Percentage"),
        category_names=category_names,
    )

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500
# ---------------- START ----------------
load_users()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
