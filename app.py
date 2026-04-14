from flask import Flask, render_template, request, redirect, url_for, send_file
from report import generate_report
import cv2
import mediapipe as mp
import os   # 🔥 مهم للـ download check

app = Flask(__name__)

# ---------------- MediaPipe ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
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

negative_questions = [3, 5, 8, 9, 12]

# ---------------- Session ----------------
session_data = {
    "age": 0,
    "gender": 0,
    "answers": [],
    "last_result": None
}

# ---------------- Eye Tracking ----------------
def run_eye_tracker():
    cap = cv2.VideoCapture(0)
    contact_frames = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            contact_frames += 1

        total_frames += 1
        cv2.imshow("Eye Test - Press Q", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return round((contact_frames/total_frames)*100 if total_frames else 0, 2)

# ---------------- Eye ----------------
def interpret_eye_contact(p):
    if p >= 70: return "Good Eye Contact"
    elif p >= 40: return "Moderate Eye Contact"
    else: return "Poor Eye Contact"

def eye_to_score(p):
    if p >= 70: return 1
    elif p >= 50: return 2
    elif p >= 30: return 3
    else: return 4

# ---------------- CARS ----------------
mapping = [
    "Relating to People","Imitation","Emotional Response","Body Use",
    "Object Use","Adaptation to Change","Visual Response",
    "Listening Response","Taste/Smell/Touch","Fear/Nervousness",
    "Verbal Communication","Non-verbal Communication",
    "Activity Level","Level of Consistency","General Impression"
]

def adjust_score(i, val):
    if i in negative_questions:
        return 5 - val
    return val

def calculate_categories(answers, eye_score):
    categories = {k: [] for k in mapping}
    categories["Visual Response"] = eye_score

    for i, cat in enumerate(mapping):
        if cat != "Visual Response":
            val = adjust_score(i, answers[i])
            categories[cat].append(val)

    for k in categories:
        if isinstance(categories[k], list):
            categories[k] = sum(categories[k]) / len(categories[k]) if categories[k] else 1

    return categories

def total_cars_score(c):
    return sum(c.values())

# ---------------- Diagnosis ----------------
def interpret(cars_score, categories, eye_percent):

    severe = sum(1 for v in categories.values() if v >= 3)
    mild = sum(1 for v in categories.values() if v >= 2.5)

    if severe >= 5:
        return "Autism"

    if mild >= 7 and eye_percent < 50:
        return "Autism"

    if cars_score < 30 and severe < 3:
        return "No Autism"

    return "At Risk (Needs Evaluation)"

# ---------------- Specialist ----------------
def generate_insights(categories, eye_percent):
    insights = []

    if eye_percent < 40:
        insights.append("Low eye contact detected")

    if categories["Relating to People"] > 2.5:
        insights.append("Weak social interaction")

    if categories["Verbal Communication"] > 2.5:
        insights.append("Communication delay")

    if categories["Body Use"] > 3:
        insights.append("Repetitive behavior detected")

    return insights

def generate_recommendations(categories, eye_percent):
    rec = []

    if eye_percent < 40:
        rec.append("Encourage eye contact through play")

    if categories["Relating to People"] > 2.5:
        rec.append("Increase social interaction")

    if categories["Verbal Communication"] > 2.5:
        rec.append("Speech therapy recommended")

    if categories["Body Use"] > 3:
        rec.append("Structured routines recommended")

    return rec

# 🔥 AI Solutions
def generate_solutions(categories, eye_percent):

    solutions = []

    if eye_percent < 40:
        solutions.append({
            "title": "Improve Eye Contact",
            "steps": [
                "Play face-to-face games",
                "Use toys near your face",
                "Call child’s name frequently"
            ]
        })

    if categories["Relating to People"] > 2.5:
        solutions.append({
            "title": "Improve Social Skills",
            "steps": [
                "Encourage group play",
                "Spend daily interaction time"
            ]
        })

    if categories["Verbal Communication"] > 2.5:
        solutions.append({
            "title": "Improve Communication",
            "steps": [
                "Use simple words",
                "Encourage imitation",
                "Speech therapy recommended"
            ]
        })

    if categories["Body Use"] > 3:
        solutions.append({
            "title": "Reduce Repetitive Behavior",
            "steps": [
                "Structured routine",
                "Use sensory toys"
            ]
        })

    if categories["Activity Level"] > 3:
        solutions.append({
            "title": "Manage Hyperactivity",
            "steps": [
                "Short instructions",
                "Physical activities"
            ]
        })

    return solutions

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/parent_info", methods=["GET","POST"])
def parent_info():
    if request.method=="POST":
        session_data["age"]=int(request.form.get("age"))
        session_data["gender"]=int(request.form.get("gender"))
        session_data["answers"]=[None]*len(questions)
        return redirect(url_for("question", q=0))
    return render_template("parent_info.html")

@app.route("/questionnaire", methods=["GET","POST"])
def question():
    q_index=int(request.args.get("q",0))

    if request.method=="POST":

        if "prev" in request.form:
            return redirect(url_for("question", q=max(0,q_index-1)))

        if "next" in request.form:
            ans=request.form.get("answer")

            if ans is None or ans=="":
                return render_template(
                    "question.html",
                    question=questions[q_index],
                    q_index=q_index,
                    total=len(questions),
                    selected=session_data["answers"][q_index],
                    error="Please select an answer"
                )

            session_data["answers"][q_index]=int(ans)

            if q_index == len(questions)-1:
                return redirect(url_for("eye_test"))

            return redirect(url_for("question", q=q_index+1))

    return render_template(
        "question.html",
        question=questions[q_index],
        q_index=q_index,
        total=len(questions),
        selected=session_data["answers"][q_index]
    )

@app.route("/eye_test", methods=["GET","POST"])
def eye_test():

    if request.method=="POST":

        eye_percent = run_eye_tracker()
        eye_label = interpret_eye_contact(eye_percent)
        eye_score = eye_to_score(eye_percent)

        categories = calculate_categories(session_data["answers"], eye_score)
        cars_score = total_cars_score(categories)

        diagnosis = interpret(cars_score, categories, eye_percent)
        percentage = (cars_score/60)*100

        session_data["last_result"] = {
            "diagnosis": diagnosis,
            "percentage": percentage,
            "eye_percent": eye_percent,
            "cars_score": cars_score,
            "categories": categories
        }

        generate_report(session_data["last_result"])

        return render_template(
            "result.html",
            diagnosis=diagnosis,
            cars_score=round(cars_score,2),
            percentage=round(percentage,1),
            eye_percent=eye_percent,
            eye_label=eye_label,
            categories=categories
        )

    return render_template("eye_test.html")

# 🔥 DOWNLOAD FIX
@app.route("/download")
def download():
    if os.path.exists("report.pdf"):
        return send_file("report.pdf", as_attachment=True)
    return "PDF not found ❌"

@app.route("/specialist")
def specialist():

    data = session_data.get("last_result")

    if not data:
        return render_template("no_data.html")

    insights = generate_insights(data["categories"], data["eye_percent"])
    recommendations = generate_recommendations(data["categories"], data["eye_percent"])
    solutions = generate_solutions(data["categories"], data["eye_percent"])

    return render_template(
        "specialist.html",
        diagnosis=data["diagnosis"],
        percentage=round(data["percentage"],1),
        eye_percent=data["eye_percent"],
        cars_score=data["cars_score"],
        categories=data["categories"],
        insights=insights,
        recommendations=recommendations,
        solutions=solutions
    )

if __name__=="__main__":
    app.run(debug=True)