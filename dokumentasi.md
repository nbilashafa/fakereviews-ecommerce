# Submission 1: Fake E Commerce Reviews
Nama: Nabila Shafa Oktavia

Username dicoding: nbilashafa

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Fake Reviews Dataset](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset) |
| Masalah | Masalah yang disebabkan oleh review palsu pada e commerce adalah tentang kredibilitas review yang dapat mempengaruhi keputusan calon customer untuk membeli produk yang tidak sesuai kualitasnya dengan review. Identifikasi masalah ini akan mengurangi adanya tindakan penipuan oleh seller dalam bentuk pemalsuan review sehingga konsumen tidak akan dirugikan oleh kualitas barang yang tidak sesuai dengan review.|
| Solusi Machine Learning | Machine learning dibangun menggunakan TFX (TensorFlow Extended) dengan metode LSTM. Data terlebih dahulu akan melalui proses preprocessing awal, kemudian akan masuk ke dalam rangkaian komponen TensorFlow Extended untuk pelatihan dan evaluasi model.|
| Metode Pengolahan | Data diolah dengan langkah-langkah preprocessing yang meliputi pembersihan teks, tokenisasi, dan vektorisasi menggunakan TextVectorization layer. Setelah preprocessing, data dibagi menjadi set pelatihan dan set evaluasi.|
| Arsitektur model | Model menggunakan arsitektur Bidirectional LSTM (Long Short-Term Memory) dengan lapisan embedding untuk menangani urutan teks. Arsitektur model terdiri dari: TextVectorization layer untuk tokenisasi teks, Embedding layer untuk representasi kata, Bidirectional LSTM layer untuk memahami konteks sekuensial dalam teks, dan dense layer dengan aktivasi ReLU, serta Dense output layer dengan aktivasi softmax untuk klasifikasi multi-kelas.|
| Metrik evaluasi | Metrik yang digunakan untuk mengevaluasi performa model adalah: - FalsePositives: Jumlah prediksi positif yang salah  - TruePositives: Jumlah prediksi positif yang benar - FalseNegatives: Jumlah prediksi negatif yang salah - TrueNegatives: Jumlah prediksi negatif yang benar - BinaryAccuracy: Akurasi biner, dengan threshold kinerja yang diinginkan.|
| Performa model | Performa model menunjukkan hasil sebagai berikut: - Loss: 0.0014 - Binary Accuracy: 0.9996 - Val Loss: 1.9410 - Val Binary Accuracy: 0.8765 pada epoch ke-11. Hasil ini menunjukkan bahwa model memiliki akurasi yang sangat tinggi pada data pelatihan, namun terdapat penurunan akurasi pada data validasi yang mungkin mengindikasikan overfitting. |
