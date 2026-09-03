// 1. Deklarasi Variabel Global Model
let model;

// 2. Fungsi Memuat Model
async function loadModel() {
    const resultText = document.getElementById('result');
    resultText.innerText = "Memuat model TFLite...";
    try {
        model = await tflite.loadTFLiteModel('./best.tflite');
        resultText.innerText = "Model berhasil dimuat! Silakan unggah gambar.";
        console.log("Model berhasil dimuat.");
    } catch (error) {
        console.error("Gagal memuat model:", error);
        resultText.innerText = "Gagal memuat model. Pastikan path ./best.tflite benar.";
    }
}

// Jalankan pemuatan model saat halaman dibuka
loadModel();

// 3. Konfigurasi Nama Kelas (Sesuaikan dengan kelas lesi Anda)
const classNames = [
    "Kelas 0", "Kelas 1", "Kelas 2", "Kelas 3", "Kelas 4", 
    "Kelas 5", "Kelas 6", "Kelas 7", "Kelas 8", "Kelas 9"
];

// 4. Logika Pemrosesan Gambar dan Deteksi
const imageLoader = document.getElementById('imageLoader');
imageLoader.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (event) {
        const img = new Image();
        img.onload = async function () {
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            
            // Gambar foto asli ke canvas
            ctx.drawImage(img, 0, 0);

            const resultText = document.getElementById('result');
            
            // Cek apakah model sudah selesai dimuat sebelum memproses
            if (!model) {
                resultText.innerText = "Error: Model belum siap. Tunggu sebentar lalu coba lagi.";
                return;
            }

            resultText.innerText = "Memproses gambar...";
            
            // Preprocessing Tensor YOLO [1, 3, 640, 640]
            let tensor = tf.tidy(() => {
                return tf.browser.fromPixels(img)
                    .resizeNearestNeighbor([640, 640])
                    .toFloat()
                    .div(255.0)
                    .transpose([2, 0, 1])
                    .expandDims(0);
            });

            let predictions;
            try {
                // Eksekusi Prediksi
                predictions = model.predict(tensor);
                tensor.dispose(); // Bersihkan tensor input segera

                // Ekstrak data (menggunakan await agar browser tidak freeze)
                const outputData = await predictions.data(); 
                
                const numBoxes = 8400;
                const numChannels = 14; 

                let boxes = [];
                let scores = [];
                let classIds = [];

                // Looping ekstrak Bounding Box
                for (let i = 0; i < numBoxes; i++) {
                    const xc = outputData[i];
                    const yc = outputData[numBoxes + i];
                    const w  = outputData[numBoxes * 2 + i];
                    const h  = outputData[numBoxes * 3 + i];

                    let maxProb = 0;
                    let classId = -1;
                    
                    for (let c = 0; c < (numChannels - 4); c++) {
                        const prob = outputData[numBoxes * (4 + c) + i];
                        if (prob > maxProb) {
                            maxProb = prob;
                            classId = c;
                        }
                    }

                    // Confidence Threshold
                    if (maxProb > 0.25) {
                        const x1 = Math.max(0, (xc - w / 2) * (img.width / 640));
                        const y1 = Math.max(0, (yc - h / 2) * (img.height / 640));
                        const x2 = Math.min(img.width, (xc + w / 2) * (img.width / 640));
                        const y2 = Math.min(img.height, (yc + h / 2) * (img.height / 640));

                        boxes.push([y1, x1, y2, x2]);
                        scores.push(maxProb);
                        classIds.push(classId);
                    }
                }

                // Non-Maximum Suppression (NMS)
                if (boxes.length > 0) {
                    const boxTensor = tf.tensor2d(boxes);
                    const scoreTensor = tf.tensor1d(scores);

                    const selectedIndices = await tf.image.nonMaxSuppressionAsync(
                        boxTensor, scoreTensor, 5, 0.45, 0.25
                    );
                    const indices = await selectedIndices.data();

                    // Bersihkan canvas dan gambar ulang untuk membuang artefak sebelumnya
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0);

                    // Gambar setiap box hasil NMS
                    indices.forEach(i => {
                        const [ymin, xmin, ymax, xmax] = boxes[i];
                        const score = scores[i];
                        const cls = classIds[i];
                        const className = classNames[cls] || `Lesi #${cls}`;

                        const boxWidth = xmax - xmin;
                        const boxHeight = ymax - ymin;

                        // Styling Box
                        ctx.strokeStyle = '#00FF00'; 
                        ctx.lineWidth = 3;
                        ctx.strokeRect(xmin, ymin, boxWidth, boxHeight);

                        // Styling Label
                        const label = `${className} ${(score * 100).toFixed(1)}%`;
                        ctx.font = 'bold 16px Arial';
                        
                        const textWidth = ctx.measureText(label).width;
                        const textHeight = parseInt(ctx.font, 10);
                        const textY = ymin > 25 ? ymin - 5 : ymin + 20;

                        // Background Label
                        ctx.fillStyle = '#00FF00';
                        ctx.fillRect(xmin, textY - textHeight, textWidth + 8, textHeight + 6);

                        // Teks Label
                        ctx.fillStyle = '#000000'; 
                        ctx.fillText(label, xmin + 4, textY - 2);
                    });

                    // Pembersihan memori NMS
                    boxTensor.dispose();
                    scoreTensor.dispose();
                    selectedIndices.dispose();

                    resultText.innerText = `Deteksi selesai! Ditemukan ${indices.length} objek.`;
                } else {
                    resultText.innerText = "Deteksi selesai. Tidak ada lesi ditemukan di atas threshold.";
                }

            } catch (predError) {
                console.error("Kesalahan saat prediksi:", predError);
                resultText.innerText = "Gagal menjalankan prediksi. Cek konsol.";
            } finally {
                // Pembersihan memori wajib
                if (predictions) {
                    predictions.dispose();
                }
            }
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
});
