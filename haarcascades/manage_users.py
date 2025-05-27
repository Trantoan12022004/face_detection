import cv2
import os
import argparse
import json
import shutil
from face_recognizer import FaceRecognizer

def list_users(recognizer):
    """Liệt kê tất cả người dùng trong cơ sở dữ liệu"""
    users = recognizer.users_data
    if not users:
        print("Không có người dùng nào trong hệ thống")
        return
    
    print(f"\n{'ID':<10} {'Tên':<20} {'Số ảnh':<10} {'Thời gian tạo':<20} {'Lần nhận diện gần nhất':<20}")
    print("-" * 80)
    
    for user_id, info in users.items():
        name = info.get("name", "N/A")
        created_at = info.get("created_at", "N/A")
        last_recognized = info.get("last_recognized", "Chưa bao giờ") or "Chưa bao giờ"
        
        # Đếm số ảnh của người dùng
        face_images = info.get("face_images", [])
        if not face_images and "face_image" in info:  # Hỗ trợ định dạng cũ
            face_images = [info["face_image"]]
        
        num_faces = len(face_images)
        
        print(f"{user_id:<10} {name:<20} {num_faces:<10} {created_at:<20} {last_recognized:<20}")

def add_user(recognizer, user_id, name, image_path, additional_info=None):
    """Thêm người dùng mới"""
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
        return False
    
    # Thêm người dùng
    if additional_info is None:
        additional_info = {}
    
    result = recognizer.add_user(user_id, name, image, additional_info)
    return result

def add_face(recognizer, user_id, image_path):
    """Thêm ảnh khuôn mặt mới cho người dùng đã tồn tại"""
    # Kiểm tra người dùng tồn tại
    if user_id not in recognizer.users_data:
        print(f"Không tìm thấy người dùng với ID: {user_id}")
        return False
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
        return False
    
    try:
        # Chuyển từ màu sang ảnh xám
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện khuôn mặt bằng cascade classifier
        faces = recognizer.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            print("Không phát hiện khuôn mặt trong ảnh")
            return False
        
        # Lấy thông tin người dùng hiện tại
        user_info = recognizer.users_data[user_id]
        user_dir = os.path.join(recognizer.data_dir, user_id)
        
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
            # Chuyển đổi từ định dạng cũ sang mới
            if "face_image" in user_info:
                # Di chuyển ảnh cũ vào thư mục faces nếu cần
                old_face = user_info["face_image"]
                if os.path.exists(old_face) and os.path.dirname(old_face) != faces_dir:
                    new_path = os.path.join(faces_dir, f"{user_id}_face_1.jpg")
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
        recognizer._save_users_data()
        
        # Cập nhật lại mô hình nhận diện
        print("Đang cập nhật mô hình nhận diện...")
        recognizer._update_recognition_model()
        
        print(f"Đã thêm ảnh khuôn mặt mới (số {next_index}) cho người dùng {user_id}")
        return True
        
    except Exception as e:
        print(f"Lỗi khi thêm ảnh khuôn mặt: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def list_faces(recognizer, user_id):
    """Liệt kê tất cả ảnh khuôn mặt của một người dùng"""
    if user_id not in recognizer.users_data:
        print(f"Không tìm thấy người dùng với ID: {user_id}")
        return
    
    user_info = recognizer.users_data[user_id]
    name = user_info.get("name", "N/A")
    
    # Lấy danh sách ảnh khuôn mặt
    face_images = user_info.get("face_images", [])
    if not face_images and "face_image" in user_info:  # Hỗ trợ định dạng cũ
        face_images = [user_info["face_image"]]
    
    print(f"\nDanh sách ảnh khuôn mặt của {name} (ID: {user_id}):")
    if not face_images:
        print("Không có ảnh khuôn mặt nào")
        return
    
    print(f"Tổng số: {len(face_images)} ảnh")
    for i, face_path in enumerate(face_images):
        print(f"{i+1}. {face_path}")

def remove_face(recognizer, user_id, face_index):
    """Xóa một ảnh khuôn mặt của người dùng"""
    if user_id not in recognizer.users_data:
        print(f"Không tìm thấy người dùng với ID: {user_id}")
        return False
    
    user_info = recognizer.users_data[user_id]
    
    # Lấy danh sách ảnh khuôn mặt
    face_images = user_info.get("face_images", [])
    if not face_images and "face_image" in user_info:  # Hỗ trợ định dạng cũ
        face_images = [user_info["face_image"]]
        user_info["face_images"] = face_images
        if "face_image" in user_info:
            del user_info["face_image"]
    
    # Kiểm tra chỉ số hợp lệ
    if face_index < 1 or face_index > len(face_images):
        print(f"Chỉ số ảnh không hợp lệ. Phải từ 1 đến {len(face_images)}")
        return False
    
    # Xóa ảnh
    face_path = face_images[face_index - 1]
    try:
        # Xóa tệp nếu tồn tại
        if os.path.exists(face_path):
            os.remove(face_path)
            print(f"Đã xóa tệp: {face_path}")
        
        # Xóa khỏi danh sách
        face_images.pop(face_index - 1)
        user_info["face_images"] = face_images
        
        # Lưu thông tin cập nhật
        recognizer._save_users_data()
        
        # Cập nhật lại mô hình nhận diện
        print("Đang cập nhật mô hình nhận diện...")
        recognizer._update_recognition_model()
        
        print(f"Đã xóa ảnh khuôn mặt số {face_index} của người dùng {user_id}")
        return True
    except Exception as e:
        print(f"Lỗi khi xóa ảnh khuôn mặt: {str(e)}")
        return False

def main():
    # Thư mục hiện tại và thư mục dự án
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    
    print(f"Thư mục hiện tại: {current_dir}")
    print(f"Thư mục dự án: {base_dir}")
    
    # Đường dẫn đến thư mục dữ liệu người dùng
    user_data_dir = os.path.join(base_dir, "user_data")
    os.makedirs(user_data_dir, exist_ok=True)
    
    # Đường dẫn đến file cascade
    cascade_path = os.path.join(current_dir, "haarcascade_frontalface_default.xml")
    
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Quản lý người dùng cho hệ thống nhận diện khuôn mặt')
    subparsers = parser.add_subparsers(dest='command', help='Lệnh')
    
    # Lệnh liệt kê người dùng
    list_parser = subparsers.add_parser('list', help='Liệt kê người dùng')
    
    # Lệnh thêm người dùng
    add_parser = subparsers.add_parser('add', help='Thêm người dùng mới')
    add_parser.add_argument('--id', required=True, help='ID người dùng (duy nhất)')
    add_parser.add_argument('--name', required=True, help='Tên người dùng')
    add_parser.add_argument('--image', required=True, help='Đường dẫn đến ảnh khuôn mặt')
    add_parser.add_argument('--info', help='Đường dẫn đến file JSON chứa thông tin bổ sung')
    
    # Lệnh thêm ảnh khuôn mặt cho người dùng đã tồn tại
    add_face_parser = subparsers.add_parser('add-face', help='Thêm ảnh khuôn mặt cho người dùng đã tồn tại')
    add_face_parser.add_argument('--id', required=True, help='ID người dùng')
    add_face_parser.add_argument('--image', required=True, help='Đường dẫn đến ảnh khuôn mặt mới')
    
    # Lệnh liệt kê ảnh khuôn mặt của người dùng
    list_faces_parser = subparsers.add_parser('list-faces', help='Liệt kê ảnh khuôn mặt của người dùng')
    list_faces_parser.add_argument('--id', required=True, help='ID người dùng')
    
    # Lệnh xóa ảnh khuôn mặt
    remove_face_parser = subparsers.add_parser('remove-face', help='Xóa một ảnh khuôn mặt')
    remove_face_parser.add_argument('--id', required=True, help='ID người dùng')
    remove_face_parser.add_argument('--index', type=int, required=True, help='Chỉ số ảnh khuôn mặt cần xóa (bắt đầu từ 1)')
    
    # Lệnh xóa người dùng
    delete_parser = subparsers.add_parser('delete', help='Xóa người dùng')
    delete_parser.add_argument('--id', required=True, help='ID người dùng cần xóa')
    
    # Lệnh cập nhật thông tin người dùng
    update_parser = subparsers.add_parser('update', help='Cập nhật thông tin người dùng')
    update_parser.add_argument('--id', required=True, help='ID người dùng cần cập nhật')
    update_parser.add_argument('--name', help='Tên mới')
    update_parser.add_argument('--info', help='Đường dẫn đến file JSON chứa thông tin cập nhật')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Khởi tạo bộ nhận diện
    try:
        recognizer = FaceRecognizer(cascade_path)
        print("Đã khởi tạo bộ nhận diện thành công")
    except Exception as e:
        print(f"Lỗi khi khởi tạo bộ nhận diện: {str(e)}")
        return
    
    # Xử lý các lệnh
    if args.command == 'list':
        print("Đang liệt kê người dùng...")
        list_users(recognizer)
    
    elif args.command == 'add':
        print(f"Đang thêm người dùng {args.name} (ID: {args.id})...")
        
        # Kiểm tra đường dẫn ảnh
        image_path = args.image
        if not os.path.isabs(image_path):
            # Nếu đường dẫn tương đối, thử tìm trong thư mục images
            if image_path.startswith('../'):
                # Đường dẫn tương đối với thư mục hiện tại
                image_path = os.path.normpath(os.path.join(current_dir, image_path))
            else:
                # Thử trong thư mục images
                images_dir = os.path.join(base_dir, "images")
                image_path = os.path.join(images_dir, os.path.basename(image_path))
        
        print(f"Đường dẫn ảnh: {image_path}")
        
        if not os.path.isfile(image_path):
            print(f"Lỗi: Không tìm thấy file ảnh '{image_path}'")
            return
        
        additional_info = None
        if args.info:
            try:
                with open(args.info, 'r', encoding='utf-8') as f:
                    additional_info = json.load(f)
            except Exception as e:
                print(f"Lỗi khi đọc file thông tin: {str(e)}")
                return
        
        success = add_user(recognizer, args.id, args.name, image_path, additional_info)
        if success:
            print(f"Đã thêm người dùng {args.name} (ID: {args.id}) thành công")
        else:
            print("Thêm người dùng thất bại")
    
    elif args.command == 'add-face':
        print(f"Đang thêm ảnh khuôn mặt mới cho người dùng {args.id}...")
        
        # Kiểm tra đường dẫn ảnh
        image_path = args.image
        if not os.path.isabs(image_path):
            # Nếu đường dẫn tương đối, thử tìm trong thư mục images
            if image_path.startswith('../'):
                # Đường dẫn tương đối với thư mục hiện tại
                image_path = os.path.normpath(os.path.join(current_dir, image_path))
            else:
                # Thử trong thư mục images
                images_dir = os.path.join(base_dir, "images")
                image_path = os.path.join(images_dir, os.path.basename(image_path))
        
        print(f"Đường dẫn ảnh: {image_path}")
        
        if not os.path.isfile(image_path):
            print(f"Lỗi: Không tìm thấy file ảnh '{image_path}'")
            return
        
        add_face(recognizer, args.id, image_path)
    
    elif args.command == 'list-faces':
        list_faces(recognizer, args.id)
    
    elif args.command == 'remove-face':
        remove_face(recognizer, args.id, args.index)
    
    elif args.command == 'delete':
        print(f"Đang xóa người dùng {args.id}...")
        if args.id in recognizer.users_data:
            user_name = recognizer.users_data[args.id].get("name", "")
            
            # Xóa thư mục người dùng
            user_dir = os.path.join(recognizer.data_dir, args.id)
            if os.path.exists(user_dir):
                try:
                    shutil.rmtree(user_dir)
                    print(f"Đã xóa thư mục: {user_dir}")
                except Exception as e:
                    print(f"Cảnh báo: Không thể xóa thư mục người dùng: {str(e)}")
            
            # Xóa người dùng
            del recognizer.users_data[args.id]
            recognizer._save_users_data()
            recognizer._update_recognition_model()
            print(f"Đã xóa người dùng {args.id} ({user_name})")
        else:
            print(f"Không tìm thấy người dùng với ID: {args.id}")
    
    elif args.command == 'update':
        print(f"Đang cập nhật thông tin người dùng {args.id}...")
        if args.id not in recognizer.users_data:
            print(f"Không tìm thấy người dùng với ID: {args.id}")
            return
        
        user_info = recognizer.users_data[args.id]
        old_name = user_info.get("name", "")
        
        # Cập nhật tên nếu được cung cấp
        if args.name:
            user_info["name"] = args.name
            print(f"Đã cập nhật tên từ '{old_name}' thành '{args.name}'")
        
        # Cập nhật thông tin bổ sung nếu có
        if args.info:
            try:
                with open(args.info, 'r', encoding='utf-8') as f:
                    additional_info = json.load(f)
                    for key, value in additional_info.items():
                        user_info[key] = value
                        print(f"Đã cập nhật trường '{key}' thành '{value}'")
            except Exception as e:
                print(f"Lỗi khi đọc file thông tin: {str(e)}")
                return
        
        # Lưu thông tin đã cập nhật
        recognizer._save_users_data()
        print(f"Đã cập nhật thông tin cho người dùng {args.id}")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()