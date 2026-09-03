let model = null;

const classNames = [
    "Kelas 0", "Kelas 1", "Kelas 2", "Kelas 3", "Kelas 4", 
    "Kelas 5", "Kelas 6", "Kelas 7", "Kelas 8", "Kelas 9"
];

async function loadModel() {
    const statusText = document.getElementById('status-text');
    statusText.innerText = "Memuat model AI (TFLite)...";
    try {
        model = await tflite.loadTFLiteModel('./best.tflite');
        statusText.innerText = "Model siap! Silakan unggah citra untuk analisis.";
    } catch (error) {
        console.error("Gagal memuat model:", error);
        statusText.innerText = "Gagal memuat model. Periksa koneksi atau path file best.tflite.";
    }
}
loadModel();

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
            ctx.drawImage(img, 0, 0);

            const statusText = document.getElementById('status-text');
            if (!model) {
                statusText.innerText = "Sistem belum siap. Tunggu sebentar.";
                return;
            }

            statusText.innerText = "Mengekstraksi fitur gambar...";
            
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
                predictions = model.predict(tensor);
                tensor.dispose(); 
                const outputData = await predictions.data(); 
                
                const numBoxes = 8400; 
                const numChannels = 14; 

                let boxes = [];
                let scores = [];
                let classIds = [];

                // AUTO-DETECT SKALA: Cek apakah koordinat X pada indeks 0 itu normalized (< 1.5) atau absolut
                const isNormalized = outputData[0] <= 2.0 && outputData[numBoxes] <= 2.0;

                for (let i = 0; i < numBoxes; i++) {
                    let xc = outputData[i];
                    let yc = outputData[numBoxes + i];
                    let w  = outputData[numBoxes * 2 + i];
                    let h  = outputData[numBoxes * 3 + i];

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
                        // Perhitungan Skala Dinamis
                        const scaleX = isNormalized ? img.width : (img.width / 640);
                        const scaleY = isNormalized ? img.height : (img.height / 640);

                        const x1 = Math.max(0, (xc - w / 2) * scaleX);
                        const y1 = Math.max(0, (yc - h / 2) * scaleY);
                        const x2 = Math.min(img.width, (xc + w / 2) * scaleX);
                        const y2 = Math.min(img.height, (yc + h / 2) * scaleY);

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
                    const indices = await selectedIndices.data();

                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0);

                    indices.forEach(i => {
                        const [ymin, xmin, ymax, xmax] = boxes[i];
                        const score = scores[i];
                        const cls = classIds[i];
                        const className = classNames[cls] || `Kelas ${cls}`;

                        const boxWidth = xmax - xmin;
                        const boxHeight = ymax - ymin;

                        // Pastikan konteks garis direset dan ditebalkan
                        ctx.beginPath();
                        ctx.strokeStyle = '#e76f51'; // Warna oranye aksen
                        ctx.lineWidth = Math.max(3, img.width / 200); 
                        ctx.strokeRect(xmin, ymin, boxWidth, boxHeight);

                        const label = `${className} ${(score * 100).toFixed(0)}%`;
                        const fontSize = Math.max(16, img.width / 40);
                        ctx.font = `bold ${fontSize}px sans-serif`;
                        ctx.textBaseline = 'top';
                        
                        const textWidth = ctx.measureText(label).width;
                        const textY = ymin > fontSize + 10 ? ymin - fontSize - 6 : ymin + 4;

                        ctx.fillStyle = '#e76f51';
                        ctx.fillRect(xmin, textY - 2, textWidth + 10, fontSize + 8);

                        ctx.fillStyle = '#ffffff'; 
                        ctx.fillText(label, xmin + 5, textY + 2);
                    });

                    boxTensor.dispose();
                    scoreTensor.dispose();
                    selectedIndices.dispose();

                    statusText.innerText = `Deteksi Selesai: Distribusi gambar menunjukkan ${indices.length} temuan di atas batas pengujian.`;
                } else {
                    statusText.innerText = "Deteksi Selesai: Tidak ada temuan yang melewati pengujian hipotesis (confidence > 0.25).";
                }

            } catch (predError) {
                console.error("Kesalahan inferensi:", predError);
                statusText.innerText = "Terjadi gangguan saat memproses tensor gambar.";
            } finally {
                if (predictions) {
                    predictions.dispose();
                }
            }
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
});
