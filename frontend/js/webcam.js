let video = null;
let canvas = null;
let canvasContext = null;
let streaming = false;
let stream = null;

// Khởi tạo các phần tử
document.addEventListener('DOMContentLoaded', function() {
    video = document.getElementById('webcam');
    canvas = document.getElementById('canvas');
    
    if (canvas) {
        canvasContext = canvas.getContext('2d');
    }
    
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const captureButton = document.getElementById('captureButton');
    const cameraSelect = document.getElementById('cameraSelect');
    
    // Lấy danh sách camera
    if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
        navigator.mediaDevices.enumerateDevices()
            .then(devices => {
                let cameraCount = 0;
                devices.forEach(device => {
                    if (device.kind === 'videoinput') {
                        cameraCount++;
                        const option = document.createElement('option');
                        option.value = device.deviceId;
                        option.text = `Camera ${cameraCount}`;
                        if (cameraSelect) cameraSelect.appendChild(option);
                    }
                });
            })
            .catch(err => {
                console.error('Lỗi khi liệt kê thiết bị: ', err);
            });
    }
    
    // Bắt đầu camera
    if (startButton) {
        startButton.addEventListener('click', function() {
            startCamera();
        });
    }
    
    // Dừng camera
    if (stopButton) {
        stopButton.addEventListener('click', function() {
            stopCamera();
        });
    }
    
    // Chụp ảnh
    if (captureButton) {
        captureButton.addEventListener('click', function() {
            captureImage();
        });
    }
});

// Bắt đầu camera
function startCamera() {
    if (streaming) return;
    
    const cameraSelect = document.getElementById('cameraSelect');
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const captureButton = document.getElementById('captureButton');
    const recognizeButton = document.getElementById('recognizeButton');
    
    const constraints = {
        video: {
            width: { ideal: 640 },
            height: { ideal: 480 }
        }
    };
    
    // Sử dụng camera được chọn nếu có
    if (cameraSelect && cameraSelect.value !== '0') {
        constraints.video.deviceId = { exact: cameraSelect.value };
    }
    
    navigator.mediaDevices.getUserMedia(constraints)
        .then(function(s) {
            stream = s;
            video.srcObject = stream;
            video.play();
            
            streaming = true;
            if (startButton) startButton.disabled = true;
            if (stopButton) stopButton.disabled = false;
            if (captureButton) captureButton.disabled = false;
            if (recognizeButton) recognizeButton.disabled = false;
        })
        .catch(function(err) {
            console.error('Lỗi khi bật camera: ', err);
            alert('Không thể truy cập camera. Vui lòng cho phép quyền truy cập và thử lại.');
        });
}

// Dừng camera
function stopCamera() {
    if (!streaming) return;
    
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const captureButton = document.getElementById('captureButton');
    const recognizeButton = document.getElementById('recognizeButton');
    
    if (stream) {
        stream.getTracks().forEach(track => {
            track.stop();
        });
    }
    
    video.srcObject = null;
    streaming = false;
    
    if (startButton) startButton.disabled = false;
    if (stopButton) stopButton.disabled = true;
    if (captureButton) captureButton.disabled = true;
    if (recognizeButton) recognizeButton.disabled = true;
}

// Chụp ảnh
function captureImage() {
    if (!streaming) return;
    
    // Thiết lập kích thước cho canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Vẽ frame hiện tại từ video lên canvas
    canvasContext.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Chuyển đổi canvas thành base64
    const imageData = canvas.toDataURL('image/jpeg');
    
    // Tạo sự kiện để thông báo đã chụp ảnh
    const event = new CustomEvent('imageCaptured', { 
        detail: { image: imageData } 
    });
    window.dispatchEvent(event);
    
    return imageData;
}