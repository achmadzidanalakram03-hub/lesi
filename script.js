const imageLoader = document.getElementById('imageLoader');

// Opsional: Ubah array ini sesuai dengan nama kelas lesi mulut pada model Anda
const classNames = [
    "Kelas 0", "Kelas 1", "Kelas 2", "Kelas 3", "Kelas 4", 
    "Kelas 5", "Kelas 6", "Kelas 7", "Kelas 8", "Kelas 9"
];

imageLoader.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return; // Menangani jika user membatalkan pilihan file

    const reader = new FileReader();
    reader.onload = function (event) {
        const img = new Image();
        img.onload = async function () {
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            const resultText = document.getElementById('result');
            resultText.innerText = "Memproses gambar...";
            
            // Preprocessing
            let tensor = tf.tidy(() => {
                return tf.browser.fromPixels(img)
                    .resizeNearestNeighbor([640, 640])
                    .toFloat()
                    .div(255.0)
                    .transpose([2, 0, 1])
                    .expandDims(0);
            });

            if (!model) {
                resultText.innerText = "Model belum dimuat!";
                tensor.dispose();
                return;
            }

            let predictions;
            try {
                // 1. Eksekusi Prediksi
                predictions = model.predict(tensor);
                tensor.dispose(); // PENTING: Bersihkan memori tensor input segera setelah diprediksi

                // 2. Ekstrak data secara Asynchronous agar browser tidak freeze
                const outputData = await predictions.data(); 
                
                const numBoxes = 8400;
                const numChannels = 14; 

                let boxes = [];
                let scores = [];
                let classIds = [];

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

                    if (maxProb > 0.25) {
                        // Batasi koordinat (clamp) agar kotak tidak melampaui batas tepi gambar
                        const x1 = Math.max(0, (xc - w / 2) * (img.width / 640));
                        const y1 = Math.max(0, (yc - h / 2) * (img.height / 640));
                        const x2 = Math.min(img.width, (xc + w / 2) * (img.width / 640));
                        const y2 = Math.min(img.height, (yc + h / 2) * (img.height / 640));

                        boxes.push([y1, x1, y2, x2]);
                        scores.push(maxProb);
                        classIds.push(classId);
                    }
                }

                if (boxes.length > 0) {
                    const boxTensor = tf.tensor2d(boxes);
                    const scoreTensor = tf.tensor1d(scores);

                    const selectedIndices = await tf.image.nonMaxSuppressionAsync(
                        boxTensor, scoreTensor, 5, 0.45, 0.25
                    );
                    const indices = await selectedIndices.data(); // Gunakan await data()

                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0);

                    indices.forEach(i => {
                        const [ymin, xmin, ymax, xmax] = boxes[i];
                        const score = scores[i];
                        const cls = classIds[i];
                        const className = classNames[cls] || `Lesi #${cls}`;

                        const boxWidth = xmax - xmin;
                        const boxHeight = ymax - ymin;

                        // Menggambar Kotak Pembatas
                        ctx.strokeStyle = '#00FF00'; // Hijau terang
                        ctx.lineWidth = 3;
                        ctx.strokeRect(xmin, ymin, boxWidth, boxHeight);

                        // Menggambar Label dengan Background agar mudah dibaca
                        const label = `${className} ${(score * 100).toFixed(1)}%`;
                        ctx.font = 'bold 16px Arial';
                        
                        const textWidth = ctx.measureText(label).width;
                        const textHeight = parseInt(ctx.font, 10);
                        const textY = ymin > 25 ? ymin - 5 : ymin + 20;

                        // Background label
                        ctx.fillStyle = '#00FF00';
                        ctx.fillRect(xmin, textY - textHeight, textWidth + 8, textHeight + 6);

                        // Teks label
                        ctx.fillStyle = '#000000'; // Hitam di atas hijau
                        ctx.fillText(label, xmin + 4, textY - 2);
                    });

                    boxTensor.dispose();
                    scoreTensor.dispose();
                    selectedIndices.dispose();

                    resultText.innerText = `Deteksi selesai! Ditemukan ${indices.length} objek.`;
                } else {
                    resultText.innerText = "Deteksi selesai. Tidak ada objek ditemukan di atas threshold.";
                }

            } catch (predError) {
                console.error("Kesalahan saat prediksi:", predError);
                resultText.innerText = "Gagal menjalankan prediksi.";
            } finally {
                // PENTING: Mencegah kebocoran memori baik prediksi sukses maupun error
                if (predictions) {
                    predictions.dispose();
                }
            }
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
});
