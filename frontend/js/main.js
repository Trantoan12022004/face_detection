document.addEventListener('DOMContentLoaded', function() {
    // Phần tử giao diện
    const recognizeButton = document.getElementById('recognizeButton');
    const showDetails = document.getElementById('showDetails');
    const confidenceThreshold = document.getElementById('confidenceThreshold');
    const confidenceValue = document.getElementById('confidenceValue');
    const resultImage = document.getElementById('resultImage');
    const recognitionResults = document.getElementById('recognition-results');
    
    // Cập nhật giá trị hiển thị của thanh trượt độ tin cậy
    if (confidenceThreshold && confidenceValue) {
        confidenceThreshold.addEventListener('input', function() {
            confidenceValue.textContent = this.value + '%';
        });
    }
    
    // Lắng nghe sự kiện chụp ảnh
    window.addEventListener('imageCaptured', function(e) {
        const imageData = e.detail.image;
        
        // Hiển thị ảnh đã chụp
        if (resultImage) {
            resultImage.src = imageData;
            resultImage.style.display = 'block';
        }
    });
    
    // Xử lý nút nhận diện
    if (recognizeButton) {
        recognizeButton.addEventListener('click', function() {
            // Chụp ảnh trước khi nhận diện
            const imageData = captureImage();
            
            // Lấy các thông số cài đặt
            const showDetailsValue = showDetails ? showDetails.checked : true;
            const confidenceValue = confidenceThreshold ? confidenceThreshold.value : 30;
            
            // Gọi API nhận diện
            fetch('http://localhost:5000/api/recognize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: imageData,
                    confidence: confidenceValue,
                    show_details: showDetailsValue
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Hiển thị ảnh kết quả
                    if (resultImage) {
                        resultImage.src = data.image;
                        resultImage.style.display = 'block';
                    }
                    
                    // Hiển thị thông tin nhận diện
                    if (recognitionResults) {
                        let resultsHTML = '<h3>Kết quả nhận diện:</h3>';
                        
                        if (data.faces.length === 0) {
                            resultsHTML += '<p>Không phát hiện khuôn mặt nào.</p>';
                        } else {
                            resultsHTML += '<ul>';
                            data.faces.forEach((face, index) => {
                                const name = face.name || 'Không xác định';
                                const confidence = face.confidence.toFixed(1);
                                
                                resultsHTML += `<li class="face-item ${face.user_id ? 'recognized' : 'unknown'}">`;
                                resultsHTML += `<strong>Khuôn mặt ${index + 1}:</strong> ${name} (${confidence}%)`;
                                
                                // Thêm thông tin chi tiết nếu có
                                if (face.user_id && face.info) {
                                    resultsHTML += '<ul class="face-details">';
                                    Object.entries(face.info).forEach(([key, value]) => {
                                        if (key !== 'name' && value) {
                                            resultsHTML += `<li>${key}: ${value}</li>`;
                                        }
                                    });
                                    resultsHTML += '</ul>';
                                }
                                
                                resultsHTML += '</li>';
                            });
                            resultsHTML += '</ul>';
                        }
                        
                        recognitionResults.innerHTML = resultsHTML;
                    }
                } else {
                    alert('Lỗi: ' + (data.error || 'Không thể nhận diện'));
                }
            })
            .catch(err => {
                console.error('Lỗi khi gọi API: ', err);
                alert('Lỗi kết nối đến máy chủ. Vui lòng thử lại sau.');
            });
        });
    }
});