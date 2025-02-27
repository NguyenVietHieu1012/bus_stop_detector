import cv2
import os
from ultralytics import YOLO

# Load mô hình YOLO
model = YOLO("../models/trained_model.pt")

# Đường dẫn chứa video
video_folder = "../reality_test/20250217-1423-1526"

# Danh sách các phần mở rộng video cần xử lý (có thể thêm bớt nếu cần)
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

# Tạo folder xuất các frame gốc nếu chưa tồn tại
export_frames_folder = "export_frames"
if not os.path.exists(export_frames_folder):
    os.makedirs(export_frames_folder)

# Lặp qua tất cả các file trong thư mục video
for video_file in os.listdir(video_folder):
    if not video_file.lower().endswith(video_extensions):
        continue

    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không mở được video: {video_path}")
        continue

    frame_counter = 0
    print(f"Đang xử lý video: {video_file}")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_counter += 1
        # Infer với mô hình
        results = model(frame)
        boxes = results[0].boxes
        has_detection = False

        if boxes is not None and len(boxes) > 0:
            bboxes = boxes.xyxy.cpu().numpy()   # numpy array với shape [N, 4]
            confs = boxes.conf.cpu().numpy()      # numpy array chứa conf với shape [N]
            # Duyệt qua từng dự đoán, nếu có bất kỳ conf > 0.7 thì đánh dấu
            for bbox, conf in zip(bboxes, confs):
                if conf > 0.7:
                    has_detection = True
                    break

        # Nếu có dự đoán với conf > 0.7, lưu frame gốc (không annotate)
        if has_detection:
            frame_output_path = os.path.join(export_frames_folder,
                                             f"{os.path.splitext(video_file)[0]}_frame_{frame_counter:05d}.jpg")
            cv2.imwrite(frame_output_path, frame)

        # Hiển thị frame để theo dõi (có thể bỏ nếu không cần)
        cv2.imshow("YOLO Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

cv2.destroyAllWindows()
