import cv2
import argparse
import os
from face_detector import FaceDetector

def main():
    # Đường dẫn thư mục hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))  # thư mục haarcascades
    
    # Đường dẫn thư mục dự án (đi lên 1 cấp từ haarcascades)
    base_dir = os.path.dirname(current_dir)
    
    # Đường dẫn thư mục ảnh và kết quả
    images_dir = os.path.join(base_dir, "images")
    results_dir = os.path.join(base_dir, "results")
    
    # Hiển thị đường dẫn để debug
    print(f"Thư mục hiện tại: {current_dir}")
    print(f"Thư mục gốc dự án: {base_dir}")
    print(f"Thư mục ảnh: {images_dir}")
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Nhận diện khuôn mặt từ ảnh')
    parser.add_argument('--image', required=True, help='Tên file ảnh trong thư mục images')
    parser.add_argument('--cascade', default='haarcascade_frontalface_default.xml', 
                        help='Tên file cascade trong thư mục haarcascades')
    parser.add_argument('--output', help='Tên file kết quả (mặc định: giống tên file đầu vào)')
    # Thêm tham số điều chỉnh kích thước
    parser.add_argument('--resize', type=str, 
                        help='Kích thước đầu ra mới, định dạng WIDTHxHEIGHT hoặc tỷ lệ phần trăm (vd: 800x600, 0.5)')
    parser.add_argument('--extract-faces', action='store_true',
                        help='Lưu từng khuôn mặt riêng biệt thành file riêng')
    parser.add_argument('--combine-cascades', action='store_true',
                        help='Kết hợp nhiều bộ phát hiện để tăng độ chính xác')
    
    # Thêm các tham số mới cho chế độ nhận diện người dùng
    parser.add_argument('--recognition-mode', action='store_true',
                        help='Bật chế độ nhận diện người dùng')
    parser.add_argument('--confidence', type=float, default=30.0,
                        help='Ngưỡng độ tin cậy cho nhận diện (0-100, mặc định: 30)')
    parser.add_argument('--show-details', action='store_true',
                        help='Hiển thị thông tin chi tiết của người dùng')
    
    
    args = parser.parse_args()
    
    # Xây dựng đường dẫn đầy đủ
    image_path = os.path.join(images_dir, args.image)
    cascade_path = os.path.join(current_dir, args.cascade)
    
    # Hiển thị đường dẫn file để debug
    print(f"Đường dẫn ảnh: {image_path}")
    print(f"Đường dẫn cascade: {cascade_path}")
    
    # Kiểm tra thư mục ảnh
    if not os.path.isdir(images_dir):
        print(f"Lỗi: Thư mục ảnh không tồn tại: '{images_dir}'")
        return
        
    # Liệt kê các file trong thư mục images để debug
    print("Các file trong thư mục images:")
    for file in os.listdir(images_dir):
        print(f"- {file}")
    
    # Tạo tên file kết quả mặc định nếu không cung cấp
    if args.output is None:
        filename, ext = os.path.splitext(args.image)
        output_filename = f"{filename}_detected{ext}"
    else:
        output_filename = args.output
    
    output_path = os.path.join(results_dir, output_filename)
    
    # Kiểm tra file ảnh đầu vào tồn tại
    if not os.path.isfile(image_path):
        print(f"Lỗi: Không tìm thấy file ảnh '{image_path}'")
        print(f"Hãy đặt ảnh vào thư mục: {images_dir}")
        return
    
    # Kiểm tra file cascade tồn tại
    if not os.path.isfile(cascade_path):
        print(f"Lỗi: Không tìm thấy file cascade '{cascade_path}'")
        return
    
    # Khởi tạo bộ phát hiện khuôn mặt
    try:
        if args.recognition_mode:
            print("Sử dụng chế độ nhận diện người dùng...")
            from face_recognizer import FaceRecognizer
            face_processor = FaceRecognizer(cascade_path)
        else:
            print("Sử dụng chế độ phát hiện khuôn mặt cơ bản...")
            from face_detector import FaceDetector
            face_processor = FaceDetector(cascade_path)
    except Exception as e:
        print(f"Lỗi khi tải detector: {str(e)}")
        return
    
    # Đọc ảnh đầu vào
    print(f"Đang đọc ảnh từ: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ '{image_path}'")
        return
    
    # Xử lý tham số kích thước đầu ra
    resize_output = None
    if args.resize:
        if 'x' in args.resize:
            # Định dạng WIDTHxHEIGHT
            try:
                width, height = map(int, args.resize.lower().split('x'))
                resize_output = (width, height)
                print(f"Kích thước đầu ra: {width}x{height}")
            except ValueError:
                print(f"Lỗi: Định dạng kích thước không hợp lệ. Sử dụng WIDTHxHEIGHT (vd: 800x600)")
                return
        else:
            # Định dạng tỷ lệ phần trăm
            try:
                ratio = float(args.resize)
                if 0 < ratio <= 1.0:
                    resize_output = ratio
                    print(f"Tỷ lệ đầu ra: {ratio*100}%")
                else:
                    print(f"Lỗi: Tỷ lệ phải nằm trong khoảng 0-1.0")
                    return
            except ValueError:
                print(f"Lỗi: Định dạng kích thước không hợp lệ")
                return
    
    # Xử lý ảnh dựa trên chế độ
    if args.recognition_mode:
        # Nhận diện người dùng
        recognized_faces = face_processor.recognize_face(image, args.confidence)
        image_with_faces = face_processor.draw_recognized_faces(image, recognized_faces, args.show_details)
        
        # In thông tin người được nhận diện
        print("\nKết quả nhận diện:")
        for i, face_info in enumerate(recognized_faces):
            user_id = face_info.get("user_id")
            name = face_info.get("name", "khong xac dinh")
            confidence = face_info.get("confidence", 0)
            
            if user_id:
                print(f"Khuôn mặt {i+1}: {name} (ID: {user_id}) - Độ tin cậy: {confidence:.2f}%")
                # In thông tin bổ sung nếu yêu cầu
                if args.show_details:
                    info = face_info.get("info", {})
                    for key, value in info.items():
                        if key not in ["name", "face_image"] and value:
                            print(f"  - {key}: {value}")
            else:
                print(f"Khuôn mặt {i+1}: Không nhận diện được - Độ tin cậy: {confidence:.2f}%")
        
        faces = [face_info["bbox"] for face_info in recognized_faces]
    else:
        # Phát hiện khuôn mặt cơ bản
        image_with_faces, faces = face_processor.detect_faces(image, resize_output=resize_output)
    
    # Hiển thị số lượng khuôn mặt được phát hiện
    print(f"Đã phát hiện {len(faces)} khuôn mặt")
    
    # Lưu ảnh kết quả
    cv2.imwrite(output_path, image_with_faces)
    print(f"Đã lưu ảnh kết quả tại '{output_path}'")
    
    # Cắt và lưu từng khuôn mặt riêng biệt nếu được yêu cầu
    if args.extract_faces and len(faces) > 0:
        face_images = face_processor.extract_faces(image)  # Sửa 'detector' thành 'face_processor'
        for i, face_img in enumerate(face_images):
            face_filename = f"{os.path.splitext(output_filename)[0]}_face_{i+1}.jpg"
            face_path = os.path.join(results_dir, face_filename)
            cv2.imwrite(face_path, face_img)
            print(f"Đã lưu khuôn mặt {i+1} tại '{face_path}'")
    
       # Hiển thị ảnh với kích thước hợp lý
    def resize_for_display(img, max_width=1200, max_height=800):
        """Thay đổi kích thước ảnh để vừa với màn hình"""
        # Lấy kích thước gốc
        h, w = img.shape[:2]
        
        # Nếu ảnh đã nhỏ hơn kích thước tối đa, giữ nguyên
        if w <= max_width and h <= max_height:  # Sửa 'và' thành 'and'
            return img
        
        # Tính toán tỷ lệ
        ratio_w = max_width / w
        ratio_h = max_height / h
        ratio = min(ratio_w, ratio_h)
        
        # Kích thước mới
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # Thay đổi kích thước ảnh
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Hiển thị ảnh với kích thước: {new_w}x{new_h} (gốc: {w}x{h})")
        return resized
    
    # Hiển thị ảnh đã điều chỉnh kích thước
    display_image = resize_for_display(image_with_faces)
    cv2.imshow("Ket qua nhan dien khuon mat", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()