# Image Classification Fashion MNIST Dataset with CNN TensorFlow *Edutech*

## ***Business Understanding***
Proyek ini adalah proyek *computer vision* yang bertujuan untuk membangun sebuah sistem cerdas yang dapat mengklasifikasikan berbagai jenis item pakaian secara otomatis. Dalam industri fashion dan *e-commerce*, kemampuan untuk mengkategorikan gambar produk secara cepat dan akurat sangat penting untuk manajemen inventaris, sistem rekomendasi, dan pengalaman pengguna di platform digital.

## **Permasalahan Bisnis**
Perusahaan ritel atau *e-commerce* sering kali harus memproses ribuan gambar produk setiap hari. Proses klasifikasi manual tidak hanya lambat dan memakan biaya, tetapi juga rentan terhadap kesalahan manusia (*human error*). Oleh karena itu, dibutuhkan sebuah sistem otomatis berbasis machine learning untuk melakukan klasifikasi gambar item-item fashion secara efisien, konsisten, dan akurat.


## **Cakupan Proyek**
Proyek ini mencakup keseluruhan alur kerja pengembangan model deep learning, mulai dari persiapan data, pembangunan arsitektur model, pelatihan, evaluasi, hingga implementasinya dalam sebuah aplikasi web interaktif.

### Pemahaman Data (*Data Understanding*)
Dataset yang digunakan adalah Fashion MNIST, yang merupakan dataset standar untuk masalah klasifikasi gambar. Dataset ini terdiri dari:

- 70.000 gambar *grayscale* (hitam-putih).

- Ukuran setiap gambar adalah 28x28 piksel.

- Terdapat 10 kelas item pakaian yang berbeda (T-shirt, Trouser, Pullover, dll.).

- Dataset sudah terbagi menjadi 60.000 gambar untuk pelatihan dan 10.000 gambar untuk pengujian.

### Persiapan Data (*Data Preparation*)
Data mentah diproses melalui dua langkah utama:

- Normalisasi: Nilai piksel setiap gambar, yang semula berada dalam rentang [0, 255], diubah skalanya menjadi rentang [0, 1] dengan membaginya dengan 255.0. Ini membantu mempercepat konvergensi model saat pelatihan.

- Mengubah Dimensi (Reshaping): Dimensi data gambar diubah dari (jumlah_sampel, 28, 28) menjadi (jumlah_sampel, 28, 28, 1) untuk menyesuaikan dengan format input yang dibutuhkan oleh lapisan CNN di TensorFlow, yang memerlukan adanya channel warna.


### *Machine Learning Modeling*
Model dibangun menggunakan arsitektur **Convolutional Neural Network (CNN)** dengan Keras Sequential API. Arsitekturnya terdiri dari:

- Dua pasang lapisan `Conv2D` untuk ekstraksi fitur dan `MaxPooling2D` untuk reduksi dimensi.

- Lapisan `Flatten` untuk mengubah data dari format 2D menjadi vektor 1D.

- Lapisan `Dense` dengan 128 neuron dan aktivasi `ReLU` sebagai lapisan tersembunyi (hidden layer).

- Lapisan `Dense` output dengan 10 neuron dan aktivasi `softmax` untuk menghasilkan probabilitas setiap kelas.

Model ini dikompilasi menggunakan optimizer `adam` dan fungsi loss `sparse_categorical_crossentropy` karena meliki lebih dari 2 kelas.

### *Evaluation*
Kinerja model dievaluasi menggunakan metrik `accuracy` pada data pengujian. Selain itu, kurva `accuracy` dan loss selama pelatihan dan validasi juga diplot untuk memastikan model belajar dengan baik dan tidak mengalami *overfitting* (ketika model terlalu hafal data latih dan tidak bisa generalisasi ke data baru).

### *Deployment*
Model yang telah dilatih disimpan dalam sebuah file `cnn_model.keras`. Model ini kemudian diintegrasikan ke dalam sebuah aplikasi web sederhana yang dibangun menggunakan Streamlit. Aplikasi ini memungkinkan pengguna untuk mengunggah gambar item pakaian dan mendapatkan prediksi klasifikasi dari model secara langsung.

## **Persiapan**

Sumber data pelatihan: 
Fashion MNIST Dataset via `tensorflow.keras.datasets`
```
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

*Setup environment*:
```
// virtual enviroment setup
python -m venv .env --> membuat virtual enviroment
.env\Scripts\activate --> mengaktifkan virtual enviroment
pip install -r requirements.txt --> instal requirements

// additional commad
pip list --> melihat library yang terinstal
deactivate --> mematikan virtual enviroment
Remove-Item -Recurse -Force .\.env --> menghapus virtual enviroment
```


## **Menjalankan Sistem *Machine Learning***
Aplikasi web interaktif ini memungkinkan pengguna untuk melakukan hands-on dengan model yang telah dilatih.
Cara menjalankan aplikasinya adalah sebagai berikut:

1. Masuk kedalam direktory proyek
2. Buka terminal atau command prompt, arahkan ke direktori tersebut.
3. Jalankan perintah berikut:
   ```
   streamlit run app.py
   ```
4. Aplikasi akan terbuka secara otomatis di browser.
5. Unggah gambar item pakaian melalui antarmuka dan klik tombol **Classify This Image!** untuk melihat hasilnya.


## ***Conclusion***
Proyek ini berhasil mendemonstrasikan proses end-to-end dalam membangun sistem klasifikasi gambar menggunakan Convolutional Neural Network. Model yang dikembangkan mampu mengenali 10 jenis item pakaian dari dataset Fashion MNIST dengan akurasi yang baik. Implementasi model ke dalam aplikasi Streamlit juga membuktikan bahwa model ini dapat diaplikasikan dalam skenario dunia nyata, menyediakan antarmuka yang ramah pengguna untuk berinteraksi dengan teknologi deep learning.