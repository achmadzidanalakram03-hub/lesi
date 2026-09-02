let model;

async function loadModel() {
    document.getElementById('result').innerText = "Memuat model TFLite...";
    try {
        // Memuat model dari repositori
        model = await tflite.loadModel('./best.tflite');
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
            
            // Pra-pemrosesan tensor (sesuaikan input ukuran model YOLO, misal 640x640)
            let tensor = tf.browser.fromPixels(img)
                .resizeNearestNeighbor([640, 640])
                .toFloat()
                .div(255.0)
                .expandDims(0);

            if (model) {
                const predictions = await model.predict(tensor);
                document.getElementById('result').innerText = "Deteksi selesai! Cek konsol browser untuk detail.";
                console.log(predictions);
            }
        }
        img.src = event.target.result;
    }
    reader.readAsDataURL(e.target.files[0]);
});