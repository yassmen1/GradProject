import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

df = pd.read_excel("Autism children data 200.xlsx")

df.columns = df.columns.str.strip()
df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

# تحويل الإجابات لـ scale جديد
df = df.replace({
    "Always": 1,
    "Sometimes": 2,
    "Rarely": 3,
    "Never": 4,
    "Yes": 2,
    "No": 4
})

df["Gender"] = df["Gender"].replace({
    "Male": 1,
    "Female": 0
})

df["First  Diagnosis"] = df["First  Diagnosis"].astype(str).str.strip()
df["First  Diagnosis"] = df["First  Diagnosis"].astype("category")

label_mapping = dict(enumerate(df["First  Diagnosis"].cat.categories))
df["First  Diagnosis"] = df["First  Diagnosis"].cat.codes

X = df.drop("First  Diagnosis", axis=1)
y = df["First  Diagnosis"]

# إضافة eye contact كـ feature (simulated)
X["eye_contact"] = np.random.randint(1, 5, size=len(X))

model = RandomForestClassifier()
model.fit(X, y)

print("Model trained successfully")

def predict(input_data):
    # نضيف eye contact dummy
    input_data.append(np.random.randint(1, 5))
    prediction = model.predict([input_data])[0]
    return label_mapping[prediction]