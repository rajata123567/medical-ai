import pandas as pd
import pickle
import numpy as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("drug200.csv")

print(df.head())

# =========================
# ENCODER
# =========================

le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_chol = LabelEncoder()
le_drug = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex"])
df["BP"] = le_bp.fit_transform(df["BP"])
df["Cholesterol"] = le_chol.fit_transform(df["Cholesterol"])
df["Drug"] = le_drug.fit_transform(df["Drug"])

# =========================
# FITUR & TARGET
# =========================

X = df[["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]]

y = df["Drug"]

# =========================
# SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

# =========================
# PREDIKSI
# =========================

y_pred = model.predict(X_test)

# =========================
# AKURASI
# =========================

acc = accuracy_score(y_test, y_pred)

print(f"Akurasi Model: {acc * 100:.2f}%")

# =========================
# SIMPAN MODEL
# =========================

pickle.dump(model, open("drug_model.pkl", "wb"))

# =========================
# SIMPAN ENCODER
# =========================

encoder = {
    "sex": le_sex,
    "bp": le_bp,
    "chol": le_chol,
    "drug": le_drug
}

pickle.dump(encoder, open("drug_encoder.pkl", "wb"))

print("Model berhasil disimpan!")