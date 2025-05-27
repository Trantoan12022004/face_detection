import cv2
import numpy as np

class FaceDetector:
    def __init__(self, cascade_path):
        """
        Khởi tạo FaceDetector với đường dẫn đến file XML của Haar Cascade
        
        Args:
            cascade_path: Đường dẫn đến file Haar Cascade XML
        """
        # Tải bộ phân loại Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Không thể tải cascade từ {cascade_path}")
        print(f"Đã tải thành công bộ phân loại từ {cascade_path}")
    
    def detect_faces(self, image, resize_output=None):
        """
        Phát hiện khuôn mặt trong ảnh
        
        Args:
            image: Ảnh đầu vào (BGR)
            resize_output: Kích thước đầu ra dạng (width, height) hoặc None để giữ nguyên
            
        Returns:
            image_with_faces: Ảnh với các khuôn mặt được đánh dấu
            faces: Danh sách các khuôn mặt được phát hiện (x, y, w, h)
        """
        # Chuyển sang ảnh xám để tăng hiệu suất xử lý
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện khuôn mặt
        # scaleFactor: Tỷ lệ thu nhỏ ảnh mỗi lần quét
        # minNeighbors: Số lượng hàng xóm tối thiểu để xác định đối tượng
        # minSize: Kích thước tối thiểu của khuôn mặt
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Tạo bản sao của ảnh để vẽ lên đó
        image_with_faces = image.copy()
        
        # Vẽ hình chữ nhật xung quanh các khuôn mặt
        for (x, y, w, h) in faces:
            cv2.rectangle(image_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Điều chỉnh kích thước ảnh đầu ra nếu được yêu cầu
        if resize_output is not None:
            image_with_faces = self.resize_image(image_with_faces, resize_output)
        
        return image_with_faces, faces
    
    def resize_image(self, image, size):
        """
        Thay đổi kích thước ảnh
        
        Args:
            image: Ảnh đầu vào
            size: Kích thước mới dạng (width, height) hoặc tỷ lệ phần trăm (0-1.0)
            
        Returns:
            resized_image: Ảnh đã thay đổi kích thước
        """
        # Nếu size là một số, xem nó là tỷ lệ phần trăm
        if isinstance(size, (int, float)) and size <= 1.0:
            h, w = image.shape[:2]
            new_width = int(w * size)
            new_height = int(h * size)
            new_size = (new_width, new_height)
        else:
            new_size = size
        
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return resized
    
    def count_faces(self, image):
        """
        Đếm số lượng khuôn mặt trong ảnh
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            count: Số lượng khuôn mặt
        """
        _, faces = self.detect_faces(image)
        return len(faces)
    
    def extract_faces(self, image, padding=0.2):
        """
        Cắt các khuôn mặt từ ảnh gốc
        
        Args:
            image: Ảnh đầu vào
            padding: Khoảng đệm xung quanh khuôn mặt (% của kích thước khuôn mặt)
            
        Returns:
            face_images: Danh sách các ảnh khuôn mặt đã cắt
        """
        _, faces = self.detect_faces(image)
        face_images = []
        h, w = image.shape[:2]
        
        for (x, y, fw, fh) in faces:
            # Thêm padding
            pad_w = int(fw * padding)
            pad_h = int(fh * padding)
            
            # Tính toán vùng cắt với padding
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w, x + fw + pad_w)
            y2 = min(h, y + fh + pad_h)
            
            # Cắt khuôn mặt
            face_img = image[y1:y2, x1:x2]
            face_images.append(face_img)
            
        return face_images
    