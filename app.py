from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_file
import cv2
import mediapipe as mp
import io
import json
import os
import requests
import time



from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("Add Your API"))
app = Flask(__name__)
app.secret_key = "secret123"

# ---------------- USERS ----------------
users = {}

def save_users():
    with open("users.json", "w") as f:
        json.dump(users, f)

def load_users():
    global users
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            users = json.load(f)

# ---------------- TELEGRAM ----------------
def send_alert(msg, sensor_data=None):
    token = "PUT_YOUR_TOKEN_HERE"
    chat_id = "1128124853"

    url = f"https://api.telegram.org/bot{token}/sendMessage"

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

        response = requests.post(url, data={
            "chat_id": chat_id,
            "text": full_msg
        })

        print("TELEGRAM STATUS:", response.text)

    except Exception as e:
        print("ERROR:", e)

# ---------------- SENSOR ----------------
sensor_data = {
    "hr": 0,
    "acc": 0,
    "mic": 0,
    "freq": 0,
    "alerts": [],
    "state": "Normal"
}

def analyze_sensor(hr, acc, mic):
    alerts = []

    if hr > 140:
        alerts.append("⚠️ High Heart Rate")

    if hr != 0 and hr < 50:
        alerts.append("⚠️ Low Heart Rate")

    if acc > 15:
        alerts.append("⚠️ Sudden Movement")

    if mic > 70:
        alerts.append("⚠️ Loud Crying")

    return alerts

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
    with open("sensor_log.json", "a") as f:
        f.write(json.dumps(data) + "\n")

# ---------------- SENSOR API ----------------
@app.route("/sensor", methods=["POST"])
def sensor():
    global sensor_data

    data = request.json

    hr = data.get("hr", 0)
    acc = data.get("acc", 0)
    mic = data.get("mic", 0)
    freq = data.get("freq", 0)

    alerts = analyze_sensor(hr, acc, mic)
    state = predict_state(hr, acc, mic)

    sensor_data = {
        "hr": hr,
        "acc": acc,
        "mic": mic,
        "freq": freq,
        "alerts": alerts,
        "state": state
    }

    log_data(sensor_data)
    emergency_mode(alerts)

    for a in alerts:
        send_alert(a)

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
    "Overall, does your child behave like other children of same age?"
]

negative_questions = [3,5,8,9,12]

mapping = [
"Relating","Imitation","Emotion","Body","Objects",
"Adaptation","Visual","Listening","Sensory","Fear",
"Verbal","NonVerbal","Activity","Consistency","General"
]

def eye_to_score(p):
    if p >= 70: return 1
    elif p >= 50: return 2
    elif p >= 30: return 3
    else: return 4

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
            categories[k] = sum(categories[k]) / len(categories[k]) if categories[k] else 1

    return categories

def total_score(c):
    return sum(c.values())

def final_diagnosis(categories, eye_percent):
    autism_score = 0
    adhd_score = 0
    delay_score = 0

    if categories["Relating"] > 2.5: autism_score += 2
    if categories["Verbal"] > 2.5: autism_score += 2
    if categories["NonVerbal"] > 2.5: autism_score += 2
    if eye_percent < 40: autism_score += 2

    if categories["Activity"] > 3: adhd_score += 2

    avg_score = sum(categories.values()) / len(categories)
    if avg_score > 2.5: delay_score += 3

    if autism_score >= 6: return "Autism"
    if adhd_score >= 4: return "ADHD"
    if delay_score >= 3: return "Delay"
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

    start_time = time.time()
    duration = 60

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        eye_contact = False

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            def xy(lm):
                return int(lm.x * w), int(lm.y * h)

            try:
                lpx, lpy = xy(face.landmark[468])
                rpx, rpy = xy(face.landmark[473])

                eye_contact = True

                cv2.circle(frame, (lpx, lpy), 3, (0, 0, 255), -1)
                cv2.circle(frame, (rpx, rpy), 3, (0, 0, 255), -1)

            except:
                eye_contact = False

        if eye_contact:
            contact += 1

        total += 1

        cv2.imshow("Eye Tracking", frame)

        if time.time() - start_time >= duration:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    percent = (contact / total) * 100 if total else 0
    return round(percent, 2)

# ---------------- AUTH ----------------
@app.route("/register", methods=["GET","POST"])
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
            "chat_history": []
        }

        save_users()
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
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

    # ✅ لو المستخدم عمل assessment قبل كده
    if users[user]["last_result"] is not None:
     return render_template(
        "result.html",
        diagnosis=users[user]["last_result"]["diagnosis"],
        percentage=round(users[user]["last_result"]["percentage"], 1),
        eye_percent=users[user]["last_result"]["eye_percent"],
        categories=users[user]["last_result"]["categories"],
    )

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
    )


@app.route("/eye_test", methods=["GET", "POST"])
def eye_test():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))
    if users[user]["last_result"] is not None:
     return redirect(url_for("specialist"))

    if request.method == "POST":

        eye = run_eye_tracker()
        score_eye = eye_to_score(eye)

        categories = calculate_categories(users[user]["answers"], score_eye)

        diagnosis = final_diagnosis(categories, eye)

        cars = total_score(categories)
        percent = (cars / 60) * 100

        users[user]["last_result"] = {
            "diagnosis": diagnosis,
            "percentage": percent,
            "eye_percent": eye,
            "categories": categories,
        }

        save_users()

        return render_template(
            "result.html",
            diagnosis=diagnosis,
            percentage=round(percent, 1),
            eye_percent=eye,
            categories=categories,
        )

    return render_template("eye_test.html")


# 🔥 CHAT
@app.route("/ask_ai", methods=["POST"])
def ask_ai():
    user = get_current_user()
    if not user:
        return jsonify({"answer": "Please login again"})

    question = request.json.get("question")
    data = users[user]["last_result"]

    users[user]["chat_history"].append({"role": "user", "content": question})

    q = question.lower()
    eye = data["eye_percent"]
    diagnosis = data["diagnosis"]

    if "eye" in q:
        answer = "Low eye contact" if eye < 40 else "Good eye contact"
    elif "autism" in q or "why" in q:
        answer = f"Diagnosis is {diagnosis}"
    else:
        answer = "Ask about eye contact or behavior"

    users[user]["chat_history"].append({"role": "assistant", "content": answer})

    save_users()

    return jsonify({"answer": answer})


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

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Autism Assessment Report", styles["Title"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Diagnosis: {data['diagnosis']}", styles["Normal"]))
    content.append(
        Paragraph(f"Severity: {round(data['percentage'],1)}%", styles["Normal"])
    )
    content.append(Paragraph(f"Eye Contact: {data['eye_percent']}%", styles["Normal"]))

    doc.build(content)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="report.pdf")


# ---------------- START ----------------
load_users()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
