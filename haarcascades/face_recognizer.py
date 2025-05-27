import cv2
import os
import numpy as np
import json
import pickle
from datetime import datetime

class FaceRecognizer:
    def __init__(self, face_cascade_path, data_dir="user_data"):
        """
        Khởi tạo bộ nhận diện khuôn mặt
        
        Args:
            face_cascade_path: Đường dẫn đến file cascade để phát hiện khuôn mặt
            data_dir: Thư mục lưu dữ liệu người dùng
        """
        # Thư mục dự án
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Thư mục lưu dữ liệu của người dùng
        self.data_dir = os.path.join(self.base_dir, data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Tệp cơ sở dữ liệu người dùng
        self.users_db_file = os.path.join(self.data_dir, "users_database.json")
        self.face_encodings_file = os.path.join(self.data_dir, "face_encodings.pkl")
        
        # Tải bộ phát hiện khuôn mặt
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if self.face_cascade.empty():
            raise Exception(f"Không thể tải cascade từ: {face_cascade_path}")
        
        # Tải bộ nhận diện khuôn mặt LBPH
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Khởi tạo cơ sở dữ liệu người dùng nếu chưa tồn tại
        self._init_user_database()
        
        # Nạp mô hình nhận diện nếu đã có
        self.model_file = os.path.join(self.data_dir, "face_model.yml")
        if os.path.exists(self.model_file):
            self.recognizer.read(self.model_file)
            print(f"Đã nạp mô hình nhận diện từ {self.model_file}")
        
        # Tải dữ liệu khuôn mặt
        self.users_data = self._load_users_data()
        self.face_encodings = self._load_face_encodings()

    def _init_user_database(self):
        """Khởi tạo cơ sở dữ liệu người dùng nếu chưa tồn tại"""
        if not os.path.exists(self.users_db_file):
            with open(self.users_db_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=4)
            print(f"Đã tạo file cơ sở dữ liệu người dùng: {self.users_db_file}")
    
    def _load_users_data(self):
        """Tải dữ liệu người dùng từ file JSON"""
        if os.path.exists(self.users_db_file):
            with open(self.users_db_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_users_data(self):
        """Lưu dữ liệu người dùng vào file JSON"""
        with open(self.users_db_file, 'w', encoding='utf-8') as f:
            json.dump(self.users_data, f, ensure_ascii=False, indent=4)
    
    def _load_face_encodings(self):
        """Tải các vector đặc trưng khuôn mặt"""
        if os.path.exists(self.face_encodings_file):
            with open(self.face_encodings_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _save_face_encodings(self):
        """Lưu các vector đặc trưng khuôn mặt"""
        with open(self.face_encodings_file, 'wb') as f:
            pickle.dump(self.face_encodings, f)
    
    def add_user(self, user_id, name, image, additional_info=None):
        """
        Thêm người dùng mới vào cơ sở dữ liệu
        
        Args:
            user_id: ID người dùng
            name: Tên người dùng
            image: Ảnh khuôn mặt (numpy array)
            additional_info: Thông tin bổ sung (dict)
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        # Phát hiện khuôn mặt trong ảnh
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            print("Không phát hiện khuôn mặt trong ảnh")
            return False
        
        if len(faces) > 1:
            print("Phát hiện nhiều khuôn mặt trong ảnh. Chỉ sử dụng khuôn mặt đầu tiên")
        
        # Lấy khuôn mặt đầu tiên
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # Chuẩn bị thư mục cho người dùng
        user_dir = os.path.join(self.data_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Lưu ảnh khuôn mặt
        face_filename = os.path.join(user_dir, f"{user_id}.jpg")
        cv2.imwrite(face_filename, gray[y:y+h, x:x+w])
        
        # Lưu thông tin người dùng
        if additional_info is None:
            additional_info = {}
            
        user_info = {
            "name": name,
            "face_image": face_filename,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_recognized": None,
            **additional_info
        }
        
        # Lưu vào cơ sở dữ liệu
        self.users_data[user_id] = user_info
        self._save_users_data()
        
        # Cập nhật mô hình nhận diện
        self._update_recognition_model()
        
        print(f"Đã thêm người dùng: {name} (ID: {user_id})")
        return True
    
    def _update_recognition_model(self):
        """Cập nhật mô hình nhận diện khuôn mặt"""
        faces = []
        labels = []
        label_ids = {}
        current_label = 0
        
        # Thu thập dữ liệu huấn luyện
        for user_id, user_info in self.users_data.items():
            face_path = user_info.get("face_image")
            if face_path and os.path.exists(face_path):
                face_img = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
                if face_img is not None:
                    # Gán ID số cho người dùng
                    if user_id not in label_ids:
                        label_ids[user_id] = current_label
                        current_label += 1
                    
                    # Thêm ảnh và nhãn vào tập dữ liệu
                    faces.append(face_img)
                    labels.append(label_ids[user_id])
        
        if not faces:
            print("Không có dữ liệu khuôn mặt để huấn luyện")
            return
        
        # Huấn luyện mô hình
        print(f"Huấn luyện mô hình với {len(faces)} khuôn mặt...")
        self.recognizer.train(faces, np.array(labels))
        
        # Lưu mô hình và ánh xạ nhãn
        self.recognizer.write(self.model_file)
        
        # Lưu ánh xạ giữa ID nhãn và ID người dùng
        self.face_encodings = {
            "label_to_user": {str(label): user_id for user_id, label in label_ids.items()},
            "user_to_label": {user_id: str(label) for user_id, label in label_ids.items()}
        }
        self._save_face_encodings()
        
        print(f"Đã cập nhật và lưu mô hình nhận diện")
    
    def recognize_face(self, image, confidence_threshold=70):
        """
        Nhận diện khuôn mặt trong ảnh
        
        Args:
            image: Ảnh đầu vào
            confidence_threshold: Ngưỡng độ tin cậy (càng thấp càng nghiêm ngặt)
            
        Returns:
            list: Danh sách các khuôn mặt đã được nhận diện với thông tin
        """
        # Kiểm tra xem có dữ liệu nhận diện không
        if not self.face_encodings or not os.path.exists(self.model_file):
            print("Chưa có dữ liệu nhận diện. Hãy thêm người dùng trước")
            return []
        
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện khuôn mặt
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        recognized_faces = []
        
        # Xử lý từng khuôn mặt được phát hiện
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            # Nhận diện khuôn mặt
            try:
                label, confidence = self.recognizer.predict(face_roi)
                
                # Đổi ngược độ tin cậy để dễ hiểu hơn (100% là hoàn toàn chính xác)
                confidence_score = 100 - confidence
                
                if confidence_score >= confidence_threshold:
                    # Lấy ID người dùng từ nhãn
                    user_id = self.face_encodings["label_to_user"].get(str(label))
                    
                    if user_id and user_id in self.users_data:
                        user_info = self.users_data[user_id]
                        
                        # Cập nhật thời gian nhận diện gần nhất
                        user_info["last_recognized"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self._save_users_data()
                        
                        recognized_faces.append({
                            "user_id": user_id,
                            "name": user_info["name"],
                            "confidence": confidence_score,
                            "bbox": (x, y, w, h),
                            "info": user_info
                        })
                    else:
                        print(f"Không tìm thấy thông tin cho người dùng với nhãn {label}")
                else:
                    # Khuôn mặt không khớp với độ tin cậy đủ cao
                    recognized_faces.append({
                        "user_id": None,
                        "name": "Không xác định",
                        "confidence": confidence_score,
                        "bbox": (x, y, w, h),
                        "info": {"name": "Không xác định"}
                    })
            except Exception as e:
                print(f"Lỗi khi nhận diện khuôn mặt: {str(e)}")
                recognized_faces.append({
                    "user_id": None,
                    "name": "Lỗi",
                    "confidence": 0,
                    "bbox": (x, y, w, h),
                    "info": {"name": "Lỗi nhận diện"}
                })
        
        return recognized_faces
    
    def draw_recognized_faces(self, image, recognized_faces, show_info=True):
        """
        Vẽ thông tin nhận diện lên ảnh
        
        Args:
            image: Ảnh gốc
            recognized_faces: Danh sách các khuôn mặt được nhận diện
            show_info: Hiển thị thông tin bổ sung
            
        Returns:
            numpy.ndarray: Ảnh với thông tin được vẽ
        """
        result_image = image.copy()
        
        for face_info in recognized_faces:
            x, y, w, h = face_info["bbox"]
            user_id = face_info["user_id"]
            name = face_info["name"]
            confidence = face_info["confidence"]
            
            # Khung màu xanh cho khuôn mặt được nhận diện, đỏ cho không xác định
            color = (0, 255, 0) if user_id else (0, 0, 255)
            thickness = 2
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color, thickness)
            
            # Vẽ tên và độ tin cậy
            label = f"{name} ({confidence:.1f}%)" if user_id else "Không xác định"
            font_scale = 0.6
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            
            # Khung nền cho văn bản
            cv2.rectangle(result_image, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)
            cv2.putText(result_image, label, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
            
            # Hiển thị thông tin bổ sung nếu có
            if show_info and user_id:
                info = face_info["info"]
                # Lọc và hiển thị một số thông tin quan trọng
                display_info = []
                
                if "email" in info:
                    display_info.append(f"Email: {info['email']}")
                if "phone" in info:
                    display_info.append(f"SĐT: {info['phone']}")
                if "position" in info:
                    display_info.append(f"Vị trí: {info['position']}")
                if "department" in info:
                    display_info.append(f"Phòng ban: {info['department']}")
                if "last_recognized" in info and info["last_recognized"]:
                    display_info.append(f"Lần cuối: {info['last_recognized']}")
                    
                # Vẽ thông tin bổ sung
                for i, text in enumerate(display_info):
                    y_pos = y + h + 20 + i * 20
                    cv2.putText(result_image, text, (x, y_pos), font, 0.5, color, 1)
        
        return result_image