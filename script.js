if (model) {
                try {
                    const predictions = model.predict(tensor);
                    document.getElementById('result').innerText = "Deteksi selesai! Cek konsol browser untuk detail.";
                    console.log("Hasil Prediksi:", predictions);

                    // --- TAMBAHAN POST-PROCESSING ---
                    // Mengambil data array dari tensor hasil prediksi [1, 14, 8400]
                    const outputData = predictions.dataSync();
                    const numBoxes = 8400;
                    const numChannels = 14; // Sesuaikan dengan jumlah channel model Anda (4 box + 10 kelas)

                    let boxes = [];
                    let scores = [];
                    let classIds = [];

                    for (let i = 0; i < numBoxes; i++) {
                        // Format YOLO output channel-first/flat: 
                        // x_center, y_center, width, height berada di offset 0, 8400, 16800, 25200
                        const xc = outputData[i];
                        const yc = outputData[numBoxes + i];
                        const w  = outputData[numBoxes * 2 + i];
                        const h  = outputData[numBoxes * 3 + i];

                        // Cari probabilitas kelas tertinggi dari sisa channel
                        let maxProb = 0;
                        let classId = -1;
                        for (let c = 0; c < (numChannels - 4); c++) {
                            const prob = outputData[numBoxes * (4 + c) + i];
                            if (prob > maxProb) {
                                maxProb = prob;
                                classId = c;
                            }
                        }

                        // Filter berdasarkan Confidence Threshold (misal > 0.25)
                        if (maxProb > 0.25) {
                            // Koordinat dinormalisasi ke 640x640, skala balik ke ukuran asli gambar (img.width, img.height)
                            const x1 = (xc - w / 2) * (img.width / 640);
                            const y1 = (yc - h / 2) * (img.height / 640);
                            const x2 = (xc + w / 2) * (img.width / 640);
                            const y2 = (yc + h / 2) * (img.height / 640);

                            boxes.push([y1, x1, y2, x2]); // format untuk tf.image.nonMaxSuppression [ymin, xmin, ymax, xmax]
                            scores.push(maxProb);
                            classIds.push(classId);
                        }
                    }

                    console.log(`Ditemukan ${boxes.length} kandidat objek di atas threshold.`);

                    // Bersihkan memori tensor prediksi untuk mencegah memory leak
                    tf.dispose(predictions);

                } catch (predError) {
                    console.error("Kesalahan saat prediksi:", predError);
                    document.getElementById('result').innerText = "Gagal menjalankan prediksi.";
                }
            }
