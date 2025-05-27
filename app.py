from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import json
from haarcascades.face_recognizer import FaceRecognizer

# Tạo một lớp JSONEncoder tùy chỉnh để xử lý các kiểu dữ liệu NumPy
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'dtype') and np.issubdtype(obj.dtype, np.number):
            return obj.item() if hasattr(obj, 'item') else obj
        return super(NumpyJSONEncoder, self).default(obj)

# Khởi tạo Flask app
app = Flask(__name__)
app.json_encoder = NumpyJSONEncoder  # Sử dụng JSON encoder tùy chỉnh
CORS(app)  # Cho phép truy cập từ frontend

# Thư mục hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# Khởi tạo FaceRecognizer
cascade_path = os.path.join(current_dir, 'haarcascades', 'haarcascade_frontalface_default.xml')
face_recognizer = FaceRecognizer(cascade_path)

# Giữ lại hàm convert_numpy_types để sử dụng khi cần
def convert_numpy_types(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'dtype') and np.issubdtype(obj.dtype, np.number):
        return obj.item() if hasattr(obj, 'item') else obj
    else:
        return obj

# API nhận diện khuôn mặt từ ảnh base64
@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    try:
        # Nhận dữ liệu ảnh dưới dạng base64
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "Không tìm thấy dữ liệu ảnh"}), 400
        
        # Giải mã ảnh base64
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Không thể giải mã ảnh"}), 400
        
        # Lấy ngưỡng nhận diện từ request hoặc mặc định
        confidence_threshold = float(data.get('confidence', 30.0))
        
        # Thực hiện nhận diện khuôn mặt
        recognized_faces = face_recognizer.recognize_face(image, confidence_threshold)
        
        # Kết quả nhận diện
        results = []
        for face in recognized_faces:
            # Chuyển đổi các kiểu dữ liệu numpy sang kiểu Python chuẩn
            face_info = convert_numpy_types(face)
            
            # Lọc thông tin cần thiết
            info_dict = {k: v for k, v in face_info.get("info", {}).items() 
                      if k not in ["face_image"]}
            
            results.append({
                "user_id": face_info.get("user_id"),
                "name": face_info.get("name"),
                "confidence": face_info.get("confidence"),
                "bbox": face_info.get("bbox"),
                "info": info_dict
            })
        
        # Vẽ khuôn mặt lên ảnh và trả về
        image_with_faces = face_recognizer.draw_recognized_faces(
            image, recognized_faces, show_info=data.get('show_details', True)
        )
        
        # Chuyển đổi ảnh thành base64 để trả về
        _, buffer = cv2.imencode('.jpg', image_with_faces)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Trả về kết quả
        return jsonify({
            "success": True,
            "faces": results,
            "image": f"data:image/jpeg;base64,{image_base64}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API đăng ký người dùng mới
@app.route('/api/register', methods=['POST'])
def register_user():
    try:
        # Nhận dữ liệu từ request
        data = request.get_json()
        
        if not data or 'image' not in data or 'user_id' not in data or 'name' not in data:
            return jsonify({"error": "Thiếu thông tin đăng ký"}), 400
        
        # Giải mã ảnh base64
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Không thể giải mã ảnh"}), 400
        
        # Thông tin người dùng
        user_id = data['user_id']
        name = data['name']
        
        # Thông tin bổ sung (nếu có)
        additional_info = {k: v for k, v in data.items() 
                           if k not in ['image', 'user_id', 'name']}
        
        # Thêm người dùng mới
        success = face_recognizer.add_user(user_id, name, image, additional_info)
        
        if success:
            return jsonify({"success": True, "message": f"Đã đăng ký người dùng {name} thành công"})
        else:
            return jsonify({"error": "Không thể đăng ký người dùng. Vui lòng thử lại"}), 400
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API lấy danh sách người dùng
@app.route('/api/users', methods=['GET'])
def get_users():
    try:
        # Tải dữ liệu người dùng
        users_data = face_recognizer._load_users_data()
        
        # Chuyển đổi toàn bộ dữ liệu users_data sang kiểu Python chuẩn
        users_data = convert_numpy_types(users_data)
        
        # Loại bỏ đường dẫn ảnh khuôn mặt khỏi phản hồi
        for user_id, info in users_data.items():
            if 'face_image' in info:
                del info['face_image']
        
        return jsonify({"success": True, "users": users_data})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API lấy thông tin chi tiết của một người dùng
@app.route('/api/users/<user_id>', methods=['GET'])
def get_user_details(user_id):
    try:
        # Tải dữ liệu người dùng
        users_data = face_recognizer._load_users_data()
        
        if user_id not in users_data:
            return jsonify({"error": "Không tìm thấy người dùng"}), 404
        
        # Lấy thông tin người dùng
        user_info = users_data[user_id]
        user_info = convert_numpy_types(user_info)
        
        # Lấy danh sách ảnh khuôn mặt
        face_images = user_info.get("face_images", [])
        if not face_images and "face_image" in user_info:  # Hỗ trợ định dạng cũ
            face_images = [user_info["face_image"]]
        
        # Chuyển đổi ảnh khuôn mặt thành base64 để hiển thị
        face_images_base64 = []
        for face_path in face_images:
            if os.path.exists(face_path):
                try:
                    img = cv2.imread(face_path)
                    _, buffer = cv2.imencode('.jpg', img)
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    face_images_base64.append({
                        "path": face_path,
                        "data": f"data:image/jpeg;base64,{img_str}"
                    })
                except Exception as e:
                    print(f"Lỗi khi đọc ảnh {face_path}: {str(e)}")
        
        # Xóa đường dẫn ảnh từ thông tin trả về
        user_data = {k: v for k, v in user_info.items() if k not in ['face_image', 'face_images']}
        user_data['face_images'] = face_images_base64
        
        return jsonify({"success": True, "user": user_data})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API cập nhật thông tin người dùng
@app.route('/api/users/<user_id>', methods=['PUT'])
def update_user(user_id):
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Không có dữ liệu cập nhật"}), 400
        
        # Tải dữ liệu người dùng
        users_data = face_recognizer._load_users_data()
        
        if user_id not in users_data:
            return jsonify({"error": "Không tìm thấy người dùng"}), 404
        
        # Cập nhật thông tin người dùng
        user_info = users_data[user_id]
        
        # Cập nhật tên nếu có
        if 'name' in data:
            user_info['name'] = data['name']
        
        # Cập nhật các thông tin khác
        for key, value in data.items():
            if key not in ['user_id', 'face_images', 'face_image']:
                user_info[key] = value
        
        # Lưu thông tin đã cập nhật
        face_recognizer._save_users_data()
        
        return jsonify({"success": True, "message": "Đã cập nhật thông tin người dùng"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API xóa người dùng
@app.route('/api/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        # Tải dữ liệu người dùng
        users_data = face_recognizer._load_users_data()
        
        if user_id not in users_data:
            return jsonify({"error": "Không tìm thấy người dùng"}), 404
        
        # Lấy thông tin người dùng
        user_name = users_data[user_id].get("name", "")
        
        # Xóa thư mục người dùng
        user_dir = os.path.join(face_recognizer.data_dir, user_id)
        if os.path.exists(user_dir):
            import shutil
            try:
                shutil.rmtree(user_dir)
            except Exception as e:
                return jsonify({"error": f"Không thể xóa thư mục người dùng: {str(e)}"}), 500
        
        # Xóa người dùng khỏi cơ sở dữ liệu
        del users_data[user_id]
        face_recognizer._save_users_data()
        
        # Cập nhật lại mô hình nhận diện
        face_recognizer._update_recognition_model()
        
        return jsonify({"success": True, "message": f"Đã xóa người dùng {user_id} ({user_name})"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API thêm ảnh khuôn mặt cho người dùng
@app.route('/api/users/<user_id>/faces', methods=['POST'])
def add_face_image(user_id):
    try:
        # Nhận dữ liệu từ request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "Thiếu dữ liệu ảnh"}), 400
        
        # Giải mã ảnh base64
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Không thể giải mã ảnh"}), 400
        
        # Kiểm tra người dùng tồn tại
        users_data = face_recognizer._load_users_data()
        if user_id not in users_data:
            return jsonify({"error": "Không tìm thấy người dùng"}), 404
        
        # Thực hiện phát hiện khuôn mặt
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_recognizer.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return jsonify({"error": "Không phát hiện khuôn mặt trong ảnh"}), 400
        
        # Lấy thông tin người dùng hiện tại
        user_info = users_data[user_id]
        user_dir = os.path.join(face_recognizer.data_dir, user_id)
        
        # Tạo thư mục người dùng nếu chưa tồn tại
        os.makedirs(user_dir, exist_ok=True)
        
        # Chuẩn bị các thư mục cho ảnh khuôn mặt
        faces_dir = os.path.join(user_dir, "faces")
        os.makedirs(faces_dir, exist_ok=True)
        
        # Cắt khuôn mặt đầu tiên
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # Đếm số ảnh hiện tại để tạo tên file mới
        existing_faces = user_info.get("face_images", [])
        if not existing_faces and "face_image" in user_info:  # Hỗ trợ định dạng cũ
            existing_faces = [user_info["face_image"]]
            if "face_image" in user_info:
                # Di chuyển ảnh cũ vào thư mục faces nếu cần
                old_face = user_info["face_image"]
                if os.path.exists(old_face) and os.path.dirname(old_face) != faces_dir:
                    new_path = os.path.join(faces_dir, f"{user_id}_face_1.jpg")
                    import shutil
                    shutil.copy2(old_face, new_path)
                    existing_faces = [new_path]
        
        # Tạo tên file mới
        next_index = len(existing_faces) + 1
        new_face_filename = os.path.join(faces_dir, f"{user_id}_face_{next_index}.jpg")
        
        # Lưu ảnh khuôn mặt mới
        cv2.imwrite(new_face_filename, face_roi)
        
        # Cập nhật danh sách ảnh khuôn mặt
        existing_faces.append(new_face_filename)
        user_info["face_images"] = existing_faces
        
        # Xóa trường face_image cũ nếu có
        if "face_image" in user_info:
            del user_info["face_image"]
        
        # Lưu thông tin cập nhật
        face_recognizer._save_users_data()
        
        # Cập nhật lại mô hình nhận diện
        face_recognizer._update_recognition_model()
        
        # Chuyển đổi ảnh mới thành base64 để hiển thị
        _, buffer = cv2.imencode('.jpg', face_roi)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True, 
            "message": f"Đã thêm ảnh khuôn mặt mới (số {next_index}) cho người dùng {user_id}",
            "face_image": f"data:image/jpeg;base64,{img_str}",
            "face_path": new_face_filename
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# API xóa ảnh khuôn mặt
@app.route('/api/users/<user_id>/faces/<int:face_index>', methods=['DELETE'])
def delete_face_image(user_id, face_index):
    try:
        # Kiểm tra người dùng tồn tại
        users_data = face_recognizer._load_users_data()
        if user_id not in users_data:
            return jsonify({"error": "Không tìm thấy người dùng"}), 404
        
        user_info = users_data[user_id]
        
        # Lấy danh sách ảnh khuôn mặt
        face_images = user_info.get("face_images", [])
        if not face_images and "face_image" in user_info:  # Hỗ trợ định dạng cũ
            face_images = [user_info["face_image"]]
            user_info["face_images"] = face_images
            if "face_image" in user_info:
                del user_info["face_image"]
        
        # Kiểm tra chỉ số hợp lệ (chỉ số bắt đầu từ 1)
        if face_index < 1 or face_index > len(face_images):
            return jsonify({
                "error": f"Chỉ số ảnh không hợp lệ. Phải từ 1 đến {len(face_images)}"
            }), 400
        
        # Xóa ảnh
        face_path = face_images[face_index - 1]
        
        # Xóa tệp nếu tồn tại
        if os.path.exists(face_path):
            os.remove(face_path)
        
        # Xóa khỏi danh sách
        face_images.pop(face_index - 1)
        user_info["face_images"] = face_images
        
        # Lưu thông tin cập nhật
        face_recognizer._save_users_data()
        
        # Cập nhật lại mô hình nhận diện
        face_recognizer._update_recognition_model()
        
        return jsonify({
            "success": True, 
            "message": f"Đã xóa ảnh khuôn mặt số {face_index} của người dùng {user_id}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Đảm bảo thư mục lưu dữ liệu tồn tại
    os.makedirs(os.path.join(current_dir, 'haarcascades'), exist_ok=True)
    os.makedirs(os.path.join(current_dir, 'user_data'), exist_ok=True)
    
    # Sao chép FaceRecognizer vào thư mục backend/models
    os.makedirs(os.path.join(current_dir, 'models'), exist_ok=True)
    
    print("Server đang chạy tại http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)