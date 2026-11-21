# Perbandingan KNN dan Random Forest pada Klasifikasi Jenis Obat (Drug200)

Proyek ini membandingkan dua algoritma klasifikasi, **K-Nearest Neighbors (KNN)** dan **Random Forest**, pada dataset yang sama untuk memprediksi jenis obat yang sesuai bagi pasien berdasarkan karakteristik klinisnya.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626)

## Daftar Isi

- [Dataset](#dataset)
- [Struktur Repositori](#struktur-repositori)
- [Alur Kerja](#alur-kerja)
- [Hasil](#hasil)
- [Analisis Singkat](#analisis-singkat)
- [Cara Menjalankan](#cara-menjalankan)
- [Keterbatasan dan Pengembangan](#keterbatasan-dan-pengembangan)

## Dataset

Dataset **drug200.csv** berisi 200 data pasien dengan 6 kolom:

| Kolom | Keterangan |
|---|---|
| `Age` | Usia pasien (15 sampai 74 tahun) |
| `Sex` | Jenis kelamin (`F`, `M`) |
| `BP` | Tekanan darah (`HIGH`, `NORMAL`, `LOW`) |
| `Cholesterol` | Kadar kolesterol (`HIGH`, `NORMAL`) |
| `Na_to_K` | Rasio natrium terhadap kalium |
| `Drug` | **Target**: `DrugY`, `drugX`, `drugA`, `drugB`, `drugC` |

Distribusi kelas tidak seimbang: DrugY (91), drugX (54), drugA (23), drugB (16), drugC (16). Tidak ada nilai kosong maupun data duplikat.

## Struktur Repositori

```
.
├── KNN.ipynb              # Klasifikasi dengan KNN (disertai scaling)
├── Random_Forest.ipynb    # Klasifikasi dengan Random Forest
├── Dataset
│  └── drug200.csv         # Dataset (tambahkan sendiri jika belum ada)
└── README.md
```

## Alur Kerja

1. **Data cleaning**: cek nilai kosong dan duplikat.
2. **Encoding**: `LabelEncoder` untuk `Sex`, `BP`, `Cholesterol`, dan `Drug`.
3. **Split data**: 80% latih (160) dan 20% uji (40), `random_state=42`.
4. **Scaling** (khusus KNN): `MinMaxScaler` yang di-*fit* hanya pada data latih untuk mencegah *data leakage*.
5. **Modeling**:
   - KNN: `KNeighborsClassifier(n_neighbors=3)`
   - Random Forest: `RandomForestClassifier(random_state=42)`
6. **Evaluasi**: akurasi, classification report, dan confusion matrix.
7. **Pengujian**: prediksi pada data pasien baru.

## Hasil

| Model | Akurasi |
|---|---|
| KNN tanpa scaling | 78% |
| KNN dengan scaling (MinMaxScaler) | 90% |
| Random Forest (tanpa scaling) | **100%** |

> Nilai 78% adalah hasil percobaan awal sebelum scaling; notebook `KNN.ipynb` hanya menampilkan hasil setelah scaling.

**Pengujian data pasien baru** (`Age=40, Sex=M, BP=LOW, Cholesterol=HIGH, Na_to_K=4.0`): kedua model memprediksi **drugC**.

## Analisis Singkat

- **KNN sangat sensitif terhadap skala data** karena menghitung jarak antar titik. Tanpa scaling, fitur dengan rentang besar (`Age`, `Na_to_K`) mendominasi jarak, sedangkan fitur hasil encoding (0 sampai 2) hampir tidak berpengaruh. Setelah scaling, akurasi naik dari 78% menjadi 90%.
- **Random Forest tidak membutuhkan scaling** karena pohon keputusan memisahkan data berdasarkan ambang batas per fitur, bukan jarak.
- Random Forest unggul pada kasus ini, kemungkinan karena pola data bersifat berbasis aturan dan kombinasi fitur kategorikal, serta karena KNN hanya diuji dengan `k=3`.

## Cara Menjalankan

1. Clone repositori ini:

   ```bash
   git clone https://github.com/shafaayutyas/Proyek-Data-Mining.git
   cd Proyek-Data-Mining
   ```

2. Install dependensi:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. Pastikan `drug200.csv` ada di folder proyek, lalu ubah path pembacaan data di kedua notebook. Notebook asli membaca dari Google Drive (Colab):

   ```python
   # Dari:
   df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/drug200.csv')
   # Menjadi:
   df = pd.read_csv('drug200.csv')
   ```

4. Jalankan notebook:

   ```bash
   jupyter notebook
   ```

   Lalu buka `KNN.ipynb` dan `Random_Forest.ipynb`.

## Keterbatasan dan Pengembangan

- Data uji hanya 40 sampel dan evaluasi memakai satu kali split, jadi akurasi 100% belum tentu stabil. Gunakan **k-fold cross-validation** untuk hasil yang lebih andal.
- Lakukan **tuning hyperparameter** (nilai `k` pada KNN, `n_estimators` dan `max_depth` pada Random Forest).
- Gunakan `stratify=y` saat split karena kelas tidak seimbang.
- Pertimbangkan `OneHotEncoder` untuk fitur kategorikal pada KNN, karena `LabelEncoder` memberi urutan angka yang semu.
- Nilai `Na_to_K=4.0` pada data uji berada di bawah minimum data latih (6,27), jadi prediksinya bersifat indikatif.
