const imageLoader = document.getElementById('imageLoader');
imageLoader.addEventListener('change', (e) => {
    const reader = new FileReader();
    reader.onload = function (event) {
        const img = new Image();
        // Ubah fungsi onload gambar menjadi async agar bisa pakai await di dalamnya
        img.onload = async function () {
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            document.getElementById('result').innerText = "Memproses gambar...";
            
            let tensor = tf.tidy(() => {
                let imgTensor = tf.browser.fromPixels(img)
                    .resizeNearestNeighbor([640, 640])
                    .toFloat()
                    .div(255.0);
                
                return imgTensor.transpose([2, 0, 1]).expandDims(0);
            });

            if (model) {
                try {
                    const predictions = model.predict(tensor);
                    document.getElementById('result').innerText = "Deteksi selesai! Cek konsol browser untuk detail.";
                    console.log("Hasil Prediksi:", predictions);

                    const outputData = predictions.dataSync();
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
                            const x1 = (xc - w / 2) * (img.width / 640);
                            const y1 = (yc - h / 2) * (img.height / 640);
                            const x2 = (xc + w / 2) * (img.width / 640);
                            const y2 = (yc + h / 2) * (img.height / 640);

                            boxes.push([y1, x1, y2, x2]);
                            scores.push(maxProb);
                            classIds.push(classId);
                        }
                    }

                    if (boxes.length > 0) {
                        const boxTensor = tf.tensor2d(boxes);
                        const scoreTensor = tf.tensor1d(scores);

                        // Karena fungsi ini sudah async, await aman digunakan di sini
                        const selectedIndices = await tf.image.nonMaxSuppressionAsync(
                            boxTensor, scoreTensor, 5, 0.45, 0.25
                        );
                        const indices = selectedIndices.dataSync();

                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(img, 0, 0);

                        ctx.strokeStyle = '#FF0000';
                        ctx.lineWidth = 3;
                        ctx.font = '16px Arial';
                        ctx.fillStyle = '#FF0000';

                        indices.forEach(i => {
                            const [ymin, xmin, ymax, xmax] = boxes[i];
                            const score = scores[i];
                            const cls = classIds[i];

                            const boxWidth = xmax - xmin;
                            const boxHeight = ymax - ymin;

                            ctx.strokeRect(xmin, ymin, boxWidth, boxHeight);
                            const label = `Lesi #${cls} (${(score * 100).toFixed(1)}%)`;
                            ctx.fillText(label, xmin, ymin > 20 ? ymin - 5 : ymin + 20);
                        });

                        boxTensor.dispose();
                        scoreTensor.dispose();
                        selectedIndices.dispose();

                        document.getElementById('result').innerText = `Deteksi selesai! Ditemukan ${indices.length} lesi.`;
                    } else {
                        document.getElementById('result').innerText = "Deteksi selesai. Tidak ada lesi ditemukan.";
                    }

                    tf.dispose(predictions);

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
