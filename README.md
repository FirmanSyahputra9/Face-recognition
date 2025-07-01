# AI Face Recognition: Deteksi Emosi Wajah dengan HOG, Gabor, dan HOG+Gabor

Proyek ini mengimplementasikan sistem deteksi emosi wajah menggunakan fitur **HOG** (Histogram of Oriented Gradients), **Gabor**, dan kombinasi **HOG+Gabor** dengan model neural network. Sistem ini dilatih untuk mengenali tujuh emosi wajah: *angry*, *disgust*, *fear*, *happy*, *neutral*, *sad*, dan *surprise*. Proyek ini mencakup dua bagian utama: pelatihan model (`train/`) dan deteksi emosi secara real-time menggunakan webcam (`webcam.ipynb`).

## Deskripsi Proyek

Proyek ini terdiri dari dua notebook Jupyter:
1. **`train/`**: Melatih model untuk deteksi emosi menggunakan fitur HOG, Gabor, dan HOG+Gabor. Dataset yang digunakan berasal dari direktori `/kaggle/input/dataset-ekspresi/dataset-ekspresi`. Proses pelatihan melibatkan:
   - Augmentasi data (skala, rotasi, flipping, kecerahan, dan noise).
   - Ekstraksi fitur HOG dan Gabor.
   - Pelatihan model neural network dengan k-fold cross-validation.
   - Visualisasi hasil pelatihan melalui *confusion matrix* dan grafik akurasi/loss.
   - Penyimpanan model ke file `.h5`.

2. **`webcam.ipynb`**: Menggunakan model yang telah dilatih untuk mendeteksi emosi wajah secara real-time melalui webcam. Fitur utama meliputi:
   - Deteksi wajah menggunakan *Haar Cascade Classifier*.
   - Ekstraksi fitur HOG, Gabor, atau HOG+Gabor dari wajah yang terdeteksi.
   - Prediksi emosi dengan tingkat kepercayaan (*confidence score*).
   - Pilihan untuk beralih antar model (HOG, Gabor, HOG+Gabor) menggunakan tombol (`h`, `g`, `b`).

## Prasyarat

Untuk menjalankan proyek ini, Anda perlu menginstal dependensi berikut:
- Python 3.8 atau lebih tinggi
- Library Python:
  - `opencv-python`
  - `scikit-image`
  - `tensorflow`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `tqdm`

## Instalasi

1. **Clone repository**:
   ```bash
   git clone <URL_REPOSITORY>
   cd <NAMA_REPOSITORY>
   ```

2. **Instal dependensi**:
   ```bash
   pip install opencv-python scikit-image tensorflow numpy matplotlib seaborn tqdm
   ```

3. **Unduh file model `.h5`**:
   Model yang telah dilatih (`hog_emotion_model.h5`, `gabor_emotion_model.h5`, `hog_gabor_emotion_model.h5`) dapat diunduh dari:
   [Google Drive](https://drive.google.com/drive/folders/1ORDkLeLBRRKJIfGMb_HxwvvaxYTMAZ-H?hl=ID)
   Simpan file-file ini di direktori proyek Anda.

4. **Unduh notebook**:
   Pastikan Anda memiliki file `train/`, `dataset/` dan `webcam.ipynb` di direktori proyek.

## Penggunaan

### 1. Pelatihan Model
- Buka `train/` di Jupyter Notebook atau lingkungan seperti Kaggle.
- Pastikan dataset tersedia di direktori `/kaggle/input/dataset-ekspresi/dataset-ekspresi` atau sesuaikan path di kode.
- Jalankan semua sel untuk melatih model dan menyimpan file `.h5`.
- Hasil pelatihan akan menampilkan *confusion matrix* dan grafik akurasi/loss.

### 2. Deteksi Emosi dengan Webcam
- Buka `webcam.ipynb` di Jupyter Notebook.
- Pastikan file model `.h5` (`hog_emotion_model.h5`, `gabor_emotion_model.h5`, `hog_gabor_emotion_model.h5`) ada di direktori yang sama.
- Jalankan notebook untuk memulai deteksi emosi secara real-time.
- Kontrol:
  - Tekan `q` untuk keluar.
  - Tekan `h` untuk menggunakan model HOG.
  - Tekan `g` untuk menggunakan model Gabor.
  - Tekan `b` untuk menggunakan model HOG+Gabor.
- Output akan menampilkan kotak di sekitar wajah yang terdeteksi, nama emosi, dan tingkat kepercayaan (*confidence score*).

## Contoh Output
Untuk melihat contoh hasil eksekusi kode, kunjungi notebook yang telah dijalankan di Kaggle:
[Kaggle Notebook](https://www.kaggle.com/code/putekkk/notebookacf4bc6baa](https://www.kaggle.com/code/putekkk/project-hog-gabor)

## Struktur Direktori
```plaintext
├── dataset/    # Dataset yang dilatih
├── train/    # Notebook untuk melatih model
├── webcam.ipynb             # Notebook untuk deteksi emosi dengan webcam
├── hog_emotion_model.h5     # Model HOG yang telah dilatih
├── gabor_emotion_model.h5   # Model Gabor yang telah dilatih
├── hog_gabor_emotion_model.h5  # Model HOG+Gabor yang telah dilatih
├── README.md               # File ini
```

## Catatan
- Dataset ekspresi wajah tidak disertakan di repository ini. Anda dapat menggunakan dataset pada repository, menyediakan dataset sendiri atau menggunakan dataset dari Kaggle.
- Pastikan webcam terhubung dan berfungsi dengan baik untuk menjalankan `webcam.ipynb`.
- Webcam realtime hanya dapat berjalan di local (contohnya. VS Code)
- Model HOG+Gabor umumnya memberikan akurasi lebih baik karena menggabungkan fitur tepi (HOG) dan tekstur (Gabor).

## Kontribusi
Kontribusi untuk meningkatkan proyek ini sangat diterima! Silakan buat *pull request* atau laporkan *issue* di repository GitHub.

