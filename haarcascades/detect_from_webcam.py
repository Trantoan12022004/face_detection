# filepath: c:\Users\Trant\Documents\AI_python\face-detection\haarcascades\detect_from_webcam.py
import cv2
import argparse
import os
import time
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer

def list_available_cameras():
    """Liệt kê các camera khả dụng trong hệ thống"""
    available_cameras = []
    for i in range(10):  # Kiểm tra 10 camera đầu tiên
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def main():
    # Đường dẫn thư mục hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))  # thư mục haarcascades
    
    # Đường dẫn thư mục dự án (đi lên 1 cấp từ haarcascades)
    base_dir = os.path.dirname(current_dir)
    
    # Đường dẫn thư mục kết quả
    results_dir = os.path.join(base_dir, "results")
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(results_dir, exist_ok=True)
    
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Nhận diện khuôn mặt từ webcam')
    parser.add_argument('--cascade', default='haarcascade_frontalface_default.xml', 
                        help='Tên file cascade trong thư mục haarcascades')
    parser.add_argument('--camera', type=int, default=0,
                        help='Chỉ số camera (mặc định: 0)')
    parser.add_argument('--save-video', action='store_true',
                        help='Lưu video kết quả')
    parser.add_argument('--extract-faces', action='store_true',
                        help='Lưu khuôn mặt được phát hiện')
    parser.add_argument('--output', default='webcam_detection.mp4',
                        help='Tên file video kết quả (mặc định: webcam_detection.mp4)')
    parser.add_argument('--fps', type=int, default=20,
                        help='FPS cho video đầu ra (mặc định: 20)')
    parser.add_argument('--width', type=int, help='Độ phân giải chiều rộng của camera')
    parser.add_argument('--height', type=int, help='Độ phân giải chiều cao của camera')
    parser.add_argument('--brightness', type=int, help='Độ sáng của camera (-100 đến 100)')
    parser.add_argument('--contrast', type=int, help='Độ tương phản của camera (-100 đến 100)')
    parser.add_argument('--saturation', type=int, help='Độ bão hòa màu của camera (-100 đến 100)')
    parser.add_argument('--list-cameras', action='store_true', 
                        help='Liệt kê các camera khả dụng')
    parser.add_argument('--flip', type=int, choices=[0, 1, -1], help='Lật hình: 0=ngang, 1=dọc, -1=cả hai')
    parser.add_argument('--recognition-mode', action='store_true',
                        help='Bật chế độ nhận diện người dùng')
    parser.add_argument('--confidence', type=float, default=50.0,
                        help='Ngưỡng độ tin cậy cho nhận diện (0-100, mặc định: 50)')
    parser.add_argument('--show-details', action='store_true',
                        help='Hiển thị thông tin chi tiết của người dùng')
    
    args = parser.parse_args()
    
    # Liệt kê camera nếu được yêu cầu
    if args.list_cameras:
        cameras = list_available_cameras()
        if cameras:
            print("Các camera khả dụng:")
            for cam_idx in cameras:
                print(f"- Camera {cam_idx}")
        else:
            print("Không tìm thấy camera nào!")
        return
    
    # Xây dựng đường dẫn đầy đủ
    cascade_path = os.path.join(current_dir, args.cascade)
    output_path = os.path.join(results_dir, args.output)
    
    # Kiểm tra file cascade tồn tại
    if not os.path.isfile(cascade_path):
        print(f"Lỗi: Không tìm thấy file cascade '{cascade_path}'")
        return
    
    # Khởi tạo bộ phát hiện hoặc nhận diện khuôn mặt
    try:
        if args.recognition_mode:
            print("Khởi động chế độ nhận diện người dùng...")
            face_processor = FaceRecognizer(cascade_path)
        else:
            print("Khởi động chế độ phát hiện khuôn mặt cơ bản...")
            face_processor = FaceDetector(cascade_path)
    except Exception as e:
        print(f"Lỗi khi khởi tạo: {str(e)}")
        return
    
    # Mở camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở camera với chỉ số {args.camera}")
        print("Hãy thử chạy lại với tham số --list-cameras để xem các camera khả dụng")
        return
    
    # Cài đặt độ phân giải camera
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        print(f"Đã đặt độ phân giải camera: {args.width}x{args.height}")
    
    # Cài đặt các thuộc tính camera khác
    if args.brightness is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, args.brightness)
        print(f"Đã đặt độ sáng: {args.brightness}")
    
    if args.contrast is not None:
        cap.set(cv2.CAP_PROP_CONTRAST, args.contrast)
        print(f"Đã đặt độ tương phản: {args.contrast}")
    
    if args.saturation is not None:
        cap.set(cv2.CAP_PROP_SATURATION, args.saturation)
        print(f"Đã đặt độ bão hòa màu: {args.saturation}")
    
    # Lấy thông tin video sau khi đã cài đặt
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Kích thước video thực tế: {frame_width}x{frame_height}")
    
    # Chuẩn bị ghi video nếu cần
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # hoặc 'XVID' cho .avi
        video_writer = cv2.VideoWriter(output_path, fourcc, args.fps, (frame_width, frame_height))
    
    face_counter = 0  # Bộ đếm khuôn mặt
    start_time = time.time()
    frame_count = 0
    
    print("Đang chạy nhận diện từ webcam...")
    print("Nhấn 'q' để thoát, 's' để chụp ảnh")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi đọc khung hình từ camera")
            break
        
        # Lật hình nếu được yêu cầu
        if args.flip is not None:
            frame = cv2.flip(frame, args.flip)
        
        frame_count += 1
        
        # Xử lý khung hình tùy theo chế độ
        if args.recognition_mode:
            # Chế độ nhận diện người dùng
            recognized_faces = face_processor.recognize_face(frame, args.confidence)
            frame_with_faces = face_processor.draw_recognized_faces(frame, recognized_faces, args.show_details)
            num_faces = len(recognized_faces)
        else:
            # Chế độ phát hiện khuôn mặt cơ bản
            frame_with_faces, faces = face_processor.detect_faces(frame)
            num_faces = len(faces)
        
        # Tính FPS
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            cv2.putText(frame_with_faces, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Hiển thị số lượng khuôn mặt
        text = f"Faces: {num_faces}"
        cv2.putText(frame_with_faces, text, (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Thêm chỉ dẫn
        mode_text = "Chế độ: Nhận diện người dùng" if args.recognition_mode else "Chế độ: Phát hiện cơ bản"
        cv2.putText(frame_with_faces, mode_text, (10, frame_height - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Hiển thị khung hình
        window_title = "Nhận diện người dùng" if args.recognition_mode else "Phát hiện khuôn mặt"
        cv2.imshow(window_title, frame_with_faces)
        
        # Lưu video nếu cần
        if args.save_video and video_writer is not None:
            video_writer.write(frame_with_faces)
        
        # Đọc phím nhấn
        key = cv2.waitKey(1) & 0xFF
        
        # Nếu nhấn 'q', thoát
        if key == ord('q'):
            break
            
        # Nếu nhấn 's', chụp ảnh hiện tại
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            snapshot_path = os.path.join(results_dir, f"webcam_snapshot_{timestamp}.jpg")
            cv2.imwrite(snapshot_path, frame_with_faces)
            print(f"Đã lưu ảnh chụp tại '{snapshot_path}'")
            
            # Lưu khuôn mặt nếu được yêu cầu và đang ở chế độ phát hiện cơ bản
            if args.extract_faces and not args.recognition_mode:
                face_images = face_processor.extract_faces(frame)
                for i, face_img in enumerate(face_images):
                    face_filename = f"webcam_face_{timestamp}_{i+1}.jpg"
                    face_path = os.path.join(results_dir, face_filename)
                    cv2.imwrite(face_path, face_img)
                    print(f"Đã lưu khuôn mặt {i+1} tại '{face_path}'")
    
    # Giải phóng tài nguyên
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Hiển thị thống kê
    elapsed_time = time.time() - start_time
    print(f"\nTổng thời gian chạy: {elapsed_time:.2f} giây")
    print(f"Số frame đã xử lý: {frame_count}")
    if elapsed_time > 0:
        print(f"FPS trung bình: {frame_count / elapsed_time:.2f}")
    
    if args.save_video:
        print(f"Video kết quả đã được lưu tại: '{output_path}'")

if __name__ == '__main__':
    main()