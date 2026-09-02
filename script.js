let model;

async function loadModel() {
    document.getElementById('result').innerText = "Memuat model TFLite...";
    try {
        model = await tflite.loadTFLiteModel('./best.tflite');
        document.getElementById('result').innerText = "Model berhasil dimuat! Silakan unggah gambar.";
    } catch (error) {
        console.error(error);
        document.getElementById('result').innerText = "Gagal memuat model.";
    }
}

loadModel();

const imageLoader = document.getElementById('imageLoader');
imageLoader.addEventListener('change', async (e) => {
    const reader = new FileReader();
    reader.onload = async function (event) {
        const img = new Image();
        img.onload = async function () {
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            document.getElementById('result').innerText = "Memproses gambar...";
            
            // Konversi gambar dan ubah dimensi tensor dari [640, 640, 3] ke [1, 3, 640, 640]
            let tensor = tf.tidy(() => {
                let imgTensor = tf.browser.fromPixels(img)
                    .resizeNearestNeighbor([640, 640])
                    .toFloat()
                    .div(255.0);
                
                // Transpose dari [H, W, C] ke [C, H, W] lalu tambahkan batch dimension [1, C, H, W]
                return imgTensor.transpose([2, 0, 1]).expandDims(0);
            });

            if (model) {
                try {
                    const predictions = model.predict(tensor);
                    document.getElementById('result').innerText = "Deteksi selesai! Cek konsol browser untuk detail.";
                    console.log("Hasil Prediksi:", predictions);
                } catch (predError) {
                    console.error("Kesalahan saat prediksi:", predError);
                    document.getElementById('result').innerText = "Gagal menjalankan prediksi.";
                }
            }
        }
        img.src = event.target.result;
    }
    reader.readAsDataURL(e.target.files[0]);
});
