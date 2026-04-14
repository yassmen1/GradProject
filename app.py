from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_file
import cv2
import mediapipe as mp
import io
import json
import os

# 📄 PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
app.secret_key = "secret123"

# 🔥 Users DB
users = {}

# 🔥 SAVE USERS
def save_users():
    with open("users.json", "w") as f:
        json.dump(users, f)

# 🔥 LOAD USERS
def load_users():
    global users
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            users = json.load(f)

# ---------------- MediaPipe ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True
)

# ---------------- Questions ----------------
questions = [
    "Does your child respond when you call their name?",
    "Does your child imitate actions (like clapping or waving)?",
    "Does your child show appropriate emotions (smile, laugh)?",
    "Does your child show unusual body movements (rocking, spinning)?",
    "Does your child use toys in a normal way?",
    "Does your child get upset with small changes in routine?",
    "Does your child look at objects or people normally?",
    "Does your child respond to sounds or voices?",
    "Does your child react strongly to touch, taste, or smell?",
    "Does your child show unusual fears or anxiety?",
    "Does your child speak or try to communicate verbally?",
    "Does your child use eye contact or gestures to communicate?",
    "Is your child overly active or unusually inactive?",
    "Are your child's behaviors consistent day to day?",
    "Overall, does your child behave like other children of same age?"
]

negative_questions = [3,5,8,9,12]

# ---------------- Helpers ----------------
def eye_to_score(p):
    if p >= 70: return 1
    elif p >= 50: return 2
    elif p >= 30: return 3
    else: return 4

mapping = [
    "Relating","Imitation","Emotion","Body","Objects",
    "Adaptation","Visual","Listening","Sensory","Fear",
    "Verbal","NonVerbal","Activity","Consistency","General"
]

def adjust_score(i,val):
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

# 🔥 NEW DIAGNOSIS FUNCTION
def final_diagnosis(categories, eye_percent):

    autism_score = 0
    adhd_score = 0
    delay_score = 0

    # -------- Autism --------
    if categories["Relating"] > 2.5:
        autism_score += 2

    if categories["Verbal"] > 2.5:
        autism_score += 2

    if categories["NonVerbal"] > 2.5:
        autism_score += 2

    if categories["Body"] > 3:
        autism_score += 1

    if eye_percent < 40:
        autism_score += 2

    # -------- ADHD --------
    if categories["Activity"] > 3:
        adhd_score += 2

    if categories["Consistency"] > 2.5:
        adhd_score += 2

    if categories["Listening"] > 2:
        adhd_score += 1

    # -------- Developmental Delay --------
    avg_score = sum(categories.values()) / len(categories)

    if avg_score > 2.5:
        delay_score += 3

    # -------- Decision --------
    if autism_score >= 6:
        return "Autism"

    if adhd_score >= 4:
        return "ADHD"

    if delay_score >= 3:
        return "Developmental Delay"

    if autism_score >= 3:
        return "At Risk"

    return "Normal"

# ---------------- Eye Tracking ----------------
def run_eye_tracker():
    cap = cv2.VideoCapture(0)
    contact = 0
    total = 0

    import time
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
                # LEFT EYE
                l_left = face.landmark[33]
                l_right = face.landmark[133]
                l_top = face.landmark[159]
                l_bottom = face.landmark[145]
                l_pupil = face.landmark[468]

                # RIGHT EYE
                r_left = face.landmark[362]
                r_right = face.landmark[263]
                r_top = face.landmark[386]
                r_bottom = face.landmark[374]
                r_pupil = face.landmark[473]

                # coordinates
                llx, lly = xy(l_left)
                lrx, lry = xy(l_right)
                ltx, lty = xy(l_top)
                lbx, lby = xy(l_bottom)
                lpx, lpy = xy(l_pupil)

                rlx, rly = xy(r_left)
                rrx, rry = xy(r_right)
                rtx, rty = xy(r_top)
                rbx, rby = xy(r_bottom)
                rpx, rpy = xy(r_pupil)

                # centers
                lc_x = (llx + lrx) / 2
                lc_y = (lty + lby) / 2

                rc_x = (rlx + rrx) / 2
                rc_y = (rty + rby) / 2

                # 🔥 tighter thresholds (important)
                l_tx = (lrx - llx) * 0.1
                l_ty = (lby - lty) * 0.1
                r_tx = (rrx - rlx) * 0.1
                r_ty = (rby - rty) * 0.1

                left_ok = abs(lpx - lc_x) < l_tx and abs(lpy - lc_y) < l_ty
                right_ok = abs(rpx - rc_x) < r_tx and abs(rpy - rc_y) < r_ty

                if left_ok and right_ok:
                    eye_contact = True

                # رسم pupil
                cv2.circle(frame, (lpx, lpy), 3, (0, 0, 255), -1)
                cv2.circle(frame, (rpx, rpy), 3, (0, 0, 255), -1)

            except:
                # لو iris مش موجودة
                eye_contact = False

        if eye_contact:
            contact += 1

        total += 1

        # display
        color = (0,255,0) if eye_contact else (0,0,255)
        text = "Eye Contact: YES" if eye_contact else "Eye Contact: NO"

        cv2.putText(frame, text, (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Eye Tracking (Auto 60 sec)", frame)

        # ⏱️ 60 sec auto stop
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
            session.clear()
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
    u = session["user"]
    if u not in users:
        session.clear()
        return None
    return u

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    if not get_current_user():
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/parent_info", methods=["GET","POST"])
def parent_info():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    if request.method=="POST":
        users[user]["answers"]=[None]*len(questions)
        save_users()
        return redirect(url_for("question", q=0))

    return render_template("parent_info.html")


@app.route("/questionnaire", methods=["GET","POST"])
def question():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    q = int(request.args.get("q", 0))

    if request.method == "POST":

        if "prev" in request.form:
            return redirect(url_for("question", q=max(0, q-1)))

        if "next" in request.form:
            ans = request.form.get("answer")

            if not ans:
                return render_template(
                    "question.html",
                    question=questions[q],
                    q_index=q,
                    total=len(questions),
                    selected=users[user]["answers"][q],
                    error="Please select an answer first"
                )

            users[user]["answers"][q] = int(ans)
            save_users()

            if q == len(questions) - 1:
                return redirect(url_for("eye_test"))

            return redirect(url_for("question", q=q+1))

    return render_template(
        "question.html",
        question=questions[q],
        q_index=q,
        total=len(questions),
        selected=users[user]["answers"][q],
        error=None
    )


@app.route("/eye_test", methods=["GET","POST"])
def eye_test():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    if request.method=="POST":

        eye = run_eye_tracker()
        score_eye = eye_to_score(eye)

        categories = calculate_categories(users[user]["answers"], score_eye)

        diagnosis = final_diagnosis(categories, eye)

        cars = total_score(categories)
        percent = (cars/60)*100

        users[user]["last_result"] = {
            "diagnosis": diagnosis,
            "percentage": percent,
            "eye_percent": eye,
            "categories": categories
        }

        save_users()

        return render_template("result.html",
            diagnosis=diagnosis,
            percentage=round(percent,1),
            eye_percent=eye,
            categories=categories
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

    users[user]["chat_history"].append({"role":"user","content":question})

    q = question.lower()
    eye = data["eye_percent"]
    diagnosis = data["diagnosis"]

    if "eye" in q:
        answer = "Low eye contact" if eye < 40 else "Good eye contact"
    elif "autism" in q or "why" in q:
        answer = f"Diagnosis is {diagnosis}"
    else:
        answer = "Ask about eye contact or behavior"

    users[user]["chat_history"].append({"role":"assistant","content":answer})

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
    content.append(Spacer(1,10))

    content.append(Paragraph(f"Diagnosis: {data['diagnosis']}", styles["Normal"]))
    content.append(Paragraph(f"Severity: {round(data['percentage'],1)}%", styles["Normal"]))
    content.append(Paragraph(f"Eye Contact: {data['eye_percent']}%", styles["Normal"]))

    doc.build(content)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="report.pdf")


# ---------------- START ----------------
load_users()

if __name__ == "__main__":
    app.run(debug=True)