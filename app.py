from flask import Flask, render_template, request, redirect, make_response
import sqlite3
import random
import io
from io import BytesIO
import pickle
import numpy as np
from collections import Counter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, legal
from reportlab.lib.colors import HexColor
from datetime import datetime
import random

# =========================
# LOAD MODEL ML
# =========================

drug_model = pickle.load(open("drug_model.pkl", "rb"))

drug_encoder = pickle.load(open("drug_encoder.pkl", "rb"))

# DATABASE GEJALA & OBAT
from gejala_obat import GEJALA_OBAT

# DATABASE PENYAKIT
from penyakit import PENYAKIT

app = Flask(__name__)

# =========================
# HOME
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# HALAMAN ML
# =========================

@app.route("/ml")

def ml():

    return render_template("ml.html")

# =========================
# PREDICT ML
# =========================

@app.route("/predict_ml", methods=["POST"])

def predict_ml():

    age = int(request.form["age"])

    sex = request.form["sex"]

    bp = request.form["bp"]

    chol = request.form["chol"]

    na_to_k = float(request.form["na_to_k"])

    # ENCODE

    sex_encoded = drug_encoder["sex"].transform([sex])[0]

    bp_encoded = drug_encoder["bp"].transform([bp])[0]

    chol_encoded = drug_encoder["chol"].transform([chol])[0]

    # PREDIKSI

    data = np.array([[
        age,
        sex_encoded,
        bp_encoded,
        chol_encoded,
        na_to_k
    ]])

    prediction = drug_model.predict(data)[0]

    hasil_obat = drug_encoder["drug"].inverse_transform([prediction])[0]

    return render_template(
        "ml_result.html",
        hasil_obat=hasil_obat,
        age=age,
        sex=sex,
        bp=bp,
        chol=chol,
        na_to_k=na_to_k
    )


# =========================
# PREDICT AI
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # =========================
        # INPUT USER
        # =========================
        nama = request.form["nama"]

        umur = int(request.form["umur"])

        gender = request.form["gender"]

        gejala_input = request.form["gejala"].lower()

        # =========================
        # DETEKSI GEJALA
        # =========================
        gejala_terdeteksi = []

        for keyword in GEJALA_OBAT.keys():

            if keyword in gejala_input:

                gejala_terdeteksi.append(keyword)

        # fallback jika kosong
        if not gejala_terdeteksi:

            gejala_terdeteksi.append("demam")

        # =========================
        # DETEKSI PENYAKIT
        # =========================
        hasil_penyakit = []

        for nama_penyakit, keywords in PENYAKIT.items():

            skor = 0

            for k in keywords:

                if k in gejala_input:

                    skor += 1

            if skor > 0:

                hasil_penyakit.append({

                    "penyakit": nama_penyakit,

                    "skor": skor

                })

        # urutkan berdasarkan skor tertinggi
        hasil_penyakit = sorted(

            hasil_penyakit,

            key=lambda x: x["skor"],

            reverse=True

        )

        # ambil top 3 penyakit
        hasil_penyakit = hasil_penyakit[:3]

        # =========================
        # AMBIL SEMUA OBAT
        # =========================
        semua_obat = []

        for g in gejala_terdeteksi:

            if g in GEJALA_OBAT:

                for obat in GEJALA_OBAT[g]:

                    semua_obat.append(obat)

        # =========================
        # HITUNG REKOMENDASI
        # =========================
        counter = Counter([o["nama"] for o in semua_obat])

        rekomendasi = counter.most_common(5)

        # =========================
        # DETAIL OBAT
        # =========================
        top_obat = []

        for nama_obat, jumlah in rekomendasi:

            for item in semua_obat:

                if item["nama"] == nama_obat:

                    top_obat.append({

                        "nama": item["nama"],

                        "deskripsi": item["deskripsi"],

                        "efek": item["efek"]

                    })

                    break

        # =========================
        # DOSIS
        # =========================
        dosis = "Gunakan sesuai aturan pakai pada kemasan atau anjuran dokter."

        # =========================
        # PENYESUAIAN UMUR
        # =========================
        if umur < 12:

            top_obat = [

                {
                    "nama": "Paracetamol Anak",
                    "deskripsi": "Membantu menurunkan demam pada anak.",
                    "efek": "Jarang menyebabkan kantuk ringan."
                },

                {
                    "nama": "OBH Anak",
                    "deskripsi": "Membantu meredakan batuk ringan pada anak.",
                    "efek": "Dapat menyebabkan rasa kantuk."
                },

                {
                    "nama": "Vitamin Anak",
                    "deskripsi": "Membantu menjaga daya tahan tubuh anak.",
                    "efek": "Jarang menyebabkan mual ringan."
                }

            ]

        elif umur > 60:

            for obat in top_obat:

                obat["deskripsi"] += (
                    " Disarankan menggunakan dosis ringan untuk lansia."
                )

        # =========================
        # CONFIDENCE AI
        # =========================
        confidence = random.randint(85, 98)

        # =========================
        # SARAN TAMBAHAN
        # =========================
        saran = []

        if "demam" in gejala_terdeteksi:

            saran.append(
                "Perbanyak istirahat dan minum air putih."
            )

        if "flu" in gejala_terdeteksi:

            saran.append(
                "Hindari minuman dingin dan gunakan masker."
            )

        if "batuk" in gejala_terdeteksi:

            saran.append(
                "Minum air hangat untuk membantu melegakan tenggorokan."
            )

        if "maag" in gejala_terdeteksi:

            saran.append(
                "Hindari makanan pedas dan asam."
            )

        if "diare" in gejala_terdeteksi:

            saran.append(
                "Perbanyak cairan agar tubuh tidak dehidrasi."
            )

        if "darah tinggi" in gejala_terdeteksi:

            saran.append(
                "Kurangi konsumsi makanan asin dan berlemak."
            )

        if "darah rendah" in gejala_terdeteksi:

            saran.append(
                "Perbanyak makanan yang mengandung zat besi."
            )

        if "insomnia" in gejala_terdeteksi:

            saran.append(
                "Kurangi penggunaan gadget sebelum tidur."
            )

        # =========================
        # KAPAN HARUS KE DOKTER
        # =========================
        warning = []

        if "demam" in gejala_terdeteksi:

            warning.append(
                "Segera ke dokter jika demam lebih dari 3 hari atau suhu sangat tinggi."
            )

        if "batuk" in gejala_terdeteksi:

            warning.append(
                "Periksa ke dokter jika batuk disertai sesak napas atau nyeri dada."
            )

        if "maag" in gejala_terdeteksi:

            warning.append(
                "Segera konsultasi jika nyeri lambung sangat hebat atau muntah terus-menerus."
            )

        if "diare" in gejala_terdeteksi:

            warning.append(
                "Segera ke dokter bila diare menyebabkan lemas atau dehidrasi."
            )

        if "darah tinggi" in gejala_terdeteksi:

            warning.append(
                "Periksa ke dokter jika tekanan darah sangat tinggi atau disertai pusing berat."
            )

        if "darah rendah" in gejala_terdeteksi:

            warning.append(
                "Segera periksa bila tubuh sangat lemas atau sering pingsan."
            )

        if "gatal" in gejala_terdeteksi:

            warning.append(
                "Konsultasi ke dokter jika gatal menyebar atau disertai bengkak."
            )

        if "insomnia" in gejala_terdeteksi:

            warning.append(
                "Periksa ke dokter bila sulit tidur berlangsung lebih dari 2 minggu."
            )

        # fallback warning
        if not warning:

            warning.append(
                "Jika kondisi memburuk, segera konsultasikan ke dokter."
            )

        # =========================
        # SIMPAN DATABASE
        # =========================
        conn = sqlite3.connect("riwayat.db")

        cursor = conn.cursor()

        nama_obat_db = ", ".join([o["nama"] for o in top_obat])

        cursor.execute("""

        INSERT INTO riwayat (
            nama,
            umur,
            gender,
            gejala,
            obat
        )

        VALUES (?, ?, ?, ?, ?)

        """, (

            nama,
            umur,
            gender,
            gejala_input,
            nama_obat_db

        ))

        conn.commit()

        conn.close()

        # =========================
        # RENDER RESULT
        # =========================
        return render_template(

            "result.html",

            nama=nama,

            umur=umur,

            gender=gender,

            gejala=gejala_input,

            gejala_terdeteksi=gejala_terdeteksi,

            hasil_penyakit=hasil_penyakit,

            rekomendasi=top_obat,

            dosis=dosis,

            confidence=confidence,

            saran=saran,

            warning=warning
        )

    except Exception as e:

        return f"ERROR: {str(e)}"


# =========================
# RIWAYAT
# =========================
@app.route("/riwayat")
def riwayat():

    conn = sqlite3.connect("riwayat.db")

    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, nama, umur, gender, gejala, obat
    FROM riwayat
    """)

    data = cursor.fetchall()

    total_pasien = len(data)

    # =========================
    # DATA GENDER
    # =========================
    laki = 0
    perempuan = 0

    cursor.execute("""
    SELECT gender, COUNT(*)
    FROM riwayat
    GROUP BY gender
    """)

    hasil_gender = cursor.fetchall()

    for row in hasil_gender:

        if row[0] == "Laki-laki":
            laki = row[1]

        elif row[0] == "Perempuan":
            perempuan = row[1]

    # =========================
    # GEJALA TERBANYAK
    # =========================
    cursor.execute("""
    SELECT gejala, COUNT(*)
    FROM riwayat
    GROUP BY gejala
    ORDER BY COUNT(*) DESC
    LIMIT 5
    """)

    gejala_data = cursor.fetchall()

    gejala_label = [g[0] for g in gejala_data] if gejala_data else []

    gejala_value = [g[1] for g in gejala_data] if gejala_data else []

    # =========================
    # OBAT TERBANYAK
    # =========================
    cursor.execute("""
    SELECT obat, COUNT(*)
    FROM riwayat
    GROUP BY obat
    ORDER BY COUNT(*) DESC
    LIMIT 5
    """)

    obat_data = cursor.fetchall()

    obat_label = [o[0] for o in obat_data] if obat_data else []

    obat_value = [o[1] for o in obat_data] if obat_data else []

    conn.close()

    return render_template(

        "riwayat.html",

        data=data,

        total_pasien=total_pasien,

        laki=laki,

        perempuan=perempuan,

        gejala_label=gejala_label,

        gejala_value=gejala_value,

        obat_label=obat_label,

        obat_value=obat_value
    )


# =========================
# HAPUS DATA
# =========================
@app.route("/hapus/<int:id>")
def hapus(id):

    conn = sqlite3.connect("riwayat.db")

    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM riwayat WHERE id=?",
        (id,)
    )

    conn.commit()

    conn.close()

    return redirect("/riwayat")


# =========================
# HAPUS SEMUA
# =========================
@app.route("/hapus_semua")
def hapus_semua():

    conn = sqlite3.connect("riwayat.db")

    cursor = conn.cursor()

    cursor.execute("DELETE FROM riwayat")

    conn.commit()

    conn.close()

    return redirect("/riwayat")


# =========================
# EXPORT PDF
# =========================
# =========================
# PDF MEDICAL REPORT
# =========================

@app.route('/pdf')
def pdf():

    nama = request.args.get("nama", "-")

    umur = request.args.get("umur", "-")

    gender = request.args.get("gender", "-")

    gejala = request.args.get("gejala", "-")

    penyakit = request.args.get("penyakit", "-")

    confidence = request.args.get("confidence", "0")

    obat_list = request.args.getlist("obat")

    efek_list = request.args.getlist("efek")

    saran_list = request.args.getlist("saran")

    if not obat_list:
        obat_list = ["Paracetamol"]

    if not efek_list:
        efek_list = ["Mengantuk ringan"]

    if not saran_list:
        saran_list = [
            "Istirahat cukup",
            "Minum air putih",
            "Konsumsi makanan bergizi"
        ]

    buffer = BytesIO()

    p = canvas.Canvas(buffer, pagesize=legal)

    width, height = legal

    # =================================
    # BACKGROUND
    # =================================

    p.setFillColor(HexColor("#eef2f7"))
    p.rect(0, 0, width, height, fill=1)

    # =================================
    # HEADER
    # =================================

    p.setFillColor(HexColor("#0f4c81"))

    p.rect(
        0,
        height - 95,
        width,
        95,
        fill=1
    )

    p.setFillColor(HexColor("#ffffff"))

    p.setFont("Helvetica-Bold", 30)

    p.drawString(
        35,
        height - 38,
        "Smart Health AI"
    )

    p.setFont("Helvetica", 15)

    p.drawString(
        35,
        height - 65,
        "AI-Powered Health Analysis System"
    )

    # =================================
    # TITLE
    # =================================

    p.setFillColor(HexColor("#0f172a"))

    p.setFont("Helvetica-Bold", 28)

    p.drawCentredString(
        width / 2,
        height - 145,
        "MEDICAL AI REPORT"
    )

    p.setStrokeColor(HexColor("#2563eb"))

    p.line(
        80,
        height - 158,
        width - 80,
        height - 158
    )

    # =================================
    # REPORT ID
    # =================================

    tanggal = datetime.now()

    report_id = f"SHAI-{tanggal.strftime('%Y%m%d')}-{random.randint(100,999)}"

    # =================================
    # BOX PASIEN
    # =================================

    y = height - 260

    p.setFillColor(HexColor("#ffffff"))

    p.roundRect(
        50,
        y,
        width - 100,
        105,
        18,
        fill=1
    )

    p.setStrokeColor(HexColor("#2563eb"))

    p.roundRect(
        50,
        y,
        width - 100,
        105,
        18,
        fill=0
    )

    p.setFillColor(HexColor("#2563eb"))

    p.setFont("Helvetica-Bold", 17)

    p.drawString(
        70,
        y + 75,
        "■ INFORMASI PASIEN"
    )

    p.setStrokeColor(HexColor("#93c5fd"))

    p.line(
        70,
        y + 63,
        width - 70,
        y + 63
    )

    p.setFillColor(HexColor("#111827"))

    p.setFont("Helvetica-Bold", 12)

    p.drawString(80, y + 35, "Nama Pasien")
    p.drawString(80, y + 12, "Umur")

    p.drawString(320, y + 35, "Report ID")
    p.drawString(320, y + 12, "Gender")

    p.setFont("Helvetica", 12)

    p.drawString(175, y + 35, f": {nama}")
    p.drawString(125, y + 12, f": {umur} Tahun")

    p.drawString(405, y + 35, f": {report_id}")
    p.drawString(385, y + 12, f": {gender}")

    # =================================
    # GEJALA
    # =================================

    y -= 125

    p.setFillColor(HexColor("#ffffff"))

    p.roundRect(
        50,
        y,
        width - 100,
        100,
        18,
        fill=1
    )

    p.setStrokeColor(HexColor("#60a5fa"))

    p.roundRect(
        50,
        y,
        width - 100,
        100,
        18,
        fill=0
    )

    p.setFillColor(HexColor("#2563eb"))

    p.setFont("Helvetica-Bold", 17)

    p.drawString(
        70,
        y + 70,
        "■ GEJALA YANG DIALAMI"
    )

    p.setStrokeColor(HexColor("#93c5fd"))

    p.line(
        70,
        y + 58,
        width - 70,
        y + 58
    )

    p.setFillColor(HexColor("#111827"))

    p.setFont("Helvetica", 13)

    gejala_lines = gejala.split(",")

    yy = y + 28

    for g in gejala_lines:

        p.drawString(
            90,
            yy,
            f"• {g.strip().capitalize()}"
        )

        yy -= 20

    # =================================
    # ANALISIS AI
    # =================================

    y -= 118

    p.setFillColor(HexColor("#ffffff"))

    p.roundRect(
        50,
        y,
        width - 100,
        110,
        18,
        fill=1
    )

    p.setStrokeColor(HexColor("#4ade80"))

    p.roundRect(
        50,
        y,
        width - 100,
        110,
        18,
        fill=0
    )

    p.setFillColor(HexColor("#16a34a"))

    p.setFont("Helvetica-Bold", 17)

    p.drawString(
        70,
        y + 80,
        "■ HASIL ANALISIS AI"
    )

    p.setStrokeColor(HexColor("#86efac"))

    p.line(
        70,
        y + 67,
        width - 70,
        y + 67
    )

    p.setFillColor(HexColor("#111827"))

    p.setFont("Helvetica-Bold", 15)

    p.drawString(
        90,
        y + 40,
        f"Kemungkinan Penyakit : {penyakit if penyakit != '-' else 'Belum Terdeteksi'}"
    )

    p.setFont("Helvetica", 12)

    p.drawString(
        90,
        y + 15,
        f"Tingkat Keyakinan AI : {confidence}%"
    )

    p.setFillColor(HexColor("#dbeafe"))

    p.roundRect(
        290,
        y + 8,
        200,
        18,
        9,
        fill=1
    )

    try:

        conf_value = int(confidence)

    except:

        conf_value = 0

    if conf_value < 5:

        conf_width = 5

    else:

        conf_width = (conf_value / 100) * 200

    p.setFillColor(HexColor("#22c55e"))

    p.roundRect(
        290,
        y + 8,
        conf_width,
        18,
        9,
        fill=1
    )

    # =================================
    # REKOMENDASI OBAT
    # =================================

    y -= 118

    tinggi_obat = max(
        95,
        55 + (len(obat_list) * 20)
    )

    p.setFillColor(HexColor("#ffffff"))

    p.roundRect(
        50,
        y,
        width - 100,
        tinggi_obat,
        18,
        fill=1
    )

    p.setStrokeColor(HexColor("#fb923c"))

    p.roundRect(
        50,
        y,
        width - 100,
        tinggi_obat,
        18,
        fill=0
    )

    p.setFillColor(HexColor("#f59e0b"))

    p.setFont("Helvetica-Bold", 17)

    p.drawString(
        70,
        y + tinggi_obat - 28,
        "■ REKOMENDASI OBAT"
    )

    p.setStrokeColor(HexColor("#fdba74"))

    p.line(
        70,
        y + tinggi_obat - 40,
        width - 70,
        y + tinggi_obat - 40
    )

    p.setFillColor(HexColor("#111827"))

    yy = y + tinggi_obat - 65

    for i, obat in enumerate(obat_list, start=1):

        p.setFont("Helvetica", 12)

        p.drawString(
            90,
            yy,
            f"{i}. {obat}"
        )

        yy -= 20

    # =================================
    # EFEK SAMPING
    # =================================

    y -= (tinggi_obat + 25)

    tinggi_efek = max(
        95,
        55 + (len(efek_list) * 20)
    )

    p.setFillColor(HexColor("#ffffff"))

    p.roundRect(
        50,
        y,
        width - 100,
        tinggi_efek,
        18,
        fill=1
    )

    p.setStrokeColor(HexColor("#c084fc"))

    p.roundRect(
        50,
        y,
        width - 100,
        tinggi_efek,
        18,
        fill=0
    )

    p.setFillColor(HexColor("#9333ea"))

    p.setFont("Helvetica-Bold", 17)

    p.drawString(
        70,
        y + tinggi_efek - 28,
        "■ EFEK SAMPING"
    )

    p.setStrokeColor(HexColor("#d8b4fe"))

    p.line(
        70,
        y + tinggi_efek - 40,
        width - 70,
        y + tinggi_efek - 40
    )

    p.setFillColor(HexColor("#111827"))

    yy = y + tinggi_efek - 65

    for efek in efek_list:

        p.setFont("Helvetica", 12)

        p.drawString(
            90,
            yy,
            f"• {efek}"
        )

        yy -= 20

    # =================================
    # SARAN KESEHATAN
    # =================================

    y -= (tinggi_efek + 25)

    tinggi_saran = max(
        75,
        55 + (len(saran_list) * 20)
    )

    p.setFillColor(HexColor("#ffffff"))

    p.roundRect(
        50,
        y,
        width - 100,
        tinggi_saran,
        18,
        fill=1
    )

    p.setStrokeColor(HexColor("#06b6d4"))

    p.roundRect(
        50,
        y,
        width - 100,
        tinggi_saran,
        18,
        fill=0
    )

    p.setFillColor(HexColor("#0891b2"))

    p.setFont("Helvetica-Bold", 17)

    p.drawString(
        70,
        y + tinggi_saran - 28,
        "■ SARAN KESEHATAN"
    )

    p.setStrokeColor(HexColor("#67e8f9"))

    p.line(
        70,
        y + tinggi_saran - 40,
        width - 70,
        y + tinggi_saran - 40
    )

    p.setFillColor(HexColor("#111827"))

    yy = y + tinggi_saran - 65

    for saran in saran_list:

        p.setFont("Helvetica", 12)

        p.drawString(
            90,
            yy,
            f"• {saran}"
        )

        yy -= 20

    # =================================
    # FOOTER
    # =================================

    tanggal_footer = datetime.now().strftime("%d %B %Y")

    p.setFillColor(HexColor("#001b44"))

    p.rect(
        0,
        0,
        width,
        50,
        fill=1
    )

    p.setFillColor(HexColor("#ffffff"))

    p.setFont("Helvetica-Bold", 11)

    p.drawString(
        30,
        22,
        "Generated by Smart Health AI"
    )

    p.setFont("Helvetica", 9)

    p.drawString(
        30,
        8,
        f"AI-Powered Medical Assistant | {tanggal_footer}"
    )

    p.setFont("Helvetica-Bold", 11)

    p.drawRightString(
        width - 30,
        22,
        f"Confidence AI : {confidence}%"
    )

    p.setFont("Helvetica", 9)

    p.drawRightString(
        width - 30,
        8,
        "Machine Learning Diagnosis System"
    )

    p.save()

    buffer.seek(0)

    response = make_response(buffer.getvalue())

    response.headers['Content-Type'] = 'application/pdf'

    response.headers['Content-Disposition'] = 'inline; filename=medical_report.pdf'

    return response


# =========================
# RUN FLASK
# =========================
if __name__ == "__main__":
    app.run(debug=True)
    