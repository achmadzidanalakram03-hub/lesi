if (model) {
                try {
                    const predictions = model.predict(tensor);
                    document.getElementById('result').innerText = "Deteksi selesai! Cek konsol browser untuk detail.";
                    console.log("Hasil Prediksi:", predictions);

                    const outputData = predictions.dataSync();
                    const numBoxes = 8400;
                    const numChannels = 14; // Sesuaikan dengan jumlah kelas Anda (+ 4 koordinat)

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

                        // Filter Confidence Threshold awal
                        if (maxProb > 0.25) {
                            const x1 = (xc - w / 2) * (img.width / 640);
                            const y1 = (yc - h / 2) * (img.height / 640);
                            const x2 = (xc + w / 2) * (img.width / 640);
                            const y2 = (yc + h / 2) * (img.height / 640);

                            // Format [ymin, xmin, ymax, xmax] dibutuhkan oleh tf.image.nonMaxSuppression
                            boxes.push([y1, x1, y2, x2]);
                            scores.push(maxProb);
                            classIds.push(classId);
                        }
                    }

                    // Jalankan Non-Maximum Suppression (NMS) jika ada kotak yang terdeteksi
                    if (boxes.length > 0) {
                        const boxTensor = tf.tensor2d(boxes);
                        const scoreTensor = tf.tensor1d(scores);

                        // Parameter NMS: maxOutputBoxes = 5, iouThreshold = 0.45, scoreThreshold = 0.25
                        const selectedIndices = await tf.image.nonMaxSuppressionAsync(
                            boxTensor, scoreTensor, 5, 0.45, 0.25
                        );
                        const indices = selectedIndices.dataSync();

                        // Gambar ulang gambar asli di canvas untuk membersihkan jejak sebelumnya
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(img, 0, 0);

                        // Pengaturan gaya teks dan garis kotak
                        ctx.strokeStyle = '#FF0000'; // Warna merah untuk kotak
                        ctx.lineWidth = 3;
                        ctx.font = '16px Arial';
                        ctx.fillStyle = '#FF0000';

                        // Render kotak hasil NMS ke canvas
                        indices.forEach(i => {
                            const [ymin, xmin, ymax, xmax] = boxes[i];
                            const score = scores[i];
                            const cls = classIds[i];

                            const boxWidth = xmax - xmin;
                            const boxHeight = ymax - ymin;

                            // Gambar kotak pembatas
                            ctx.strokeRect(xmin, ymin, boxWidth, boxHeight);

                            // Tulis label kelas dan persentase akurasi
                            const label = `Lesi #${cls} (${(score * 100).toFixed(1)}%)`;
                            ctx.fillText(label, xmin, ymin > 20 ? ymin - 5 : ymin + 20);
                        });

                        // Bersihkan memori tensor NMS
                        boxTensor.dispose();
                        scoreTensor.dispose();
                        selectedIndices.dispose();

                        document.getElementById('result').innerText = `Deteksi selesai! Ditemukan ${indices.length} lesi.`;
                    } else {
                        document.getElementById('result').innerText = "Deteksi selesai. Tidak ada lesi yang ditemukan di atas threshold.";
                    }

                    // Bersihkan memori tensor prediksi utama
                    tf.dispose(predictions);

                } catch (predError) {
                    console.error("Kesalahan saat prediksi:", predError);
                    document.getElementById('result').innerText = "Gagal menjalankan prediksi.";
                }
            }
