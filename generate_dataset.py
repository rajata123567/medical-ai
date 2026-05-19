import csv
import random

gejala_dict = {
    "demam": ["demam", "demam tinggi", "meriang", "panas"],
    "batuk": ["batuk kering", "batuk berdahak"],
    "flu": ["flu", "pilek", "hidung tersumbat"],
    "nyeri": ["sakit kepala", "migrain", "nyeri otot"],
    "pencernaan": ["maag", "mual", "diare"]
}

obat_dict = {
    "demam": ["Paracetamol", "Ibuprofen"],
    "batuk": ["OBH Combi", "Ambroxol"],
    "flu": ["Decolgen"],
    "nyeri": ["Ibuprofen", "Paracetamol"],
    "pencernaan": ["Antasida", "Oralit", "Domperidone"]
}

dosis_list = ["3x sehari", "2x sehari", "1x sehari", "Dosis anak"]
efek_list = ["Mengantuk", "Mual", "Aman", "Pusing"]

gender_list = ["Laki-laki", "Perempuan"]

rows = []

for _ in range(300):  # 🔥 jumlah data (ubah jadi 500 juga bisa)
    kategori = random.choice(list(gejala_dict.keys()))
    gejala = random.choice(gejala_dict[kategori])
    obat = random.choice(obat_dict[kategori])
    umur = random.randint(5, 60)
    gender = random.choice(gender_list)
    dosis = random.choice(dosis_list)
    efek = random.choice(efek_list)

    rows.append([gejala, umur, gender, obat, dosis, efek])

# save
with open("data_obat.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["gejala", "umur", "gender", "obat", "dosis", "efek"])
    writer.writerows(rows)

print("Dataset berhasil dibuat!")