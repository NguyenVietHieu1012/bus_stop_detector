import cv2
import os
from ultralytics import YOLO

# Load mô hình YOLO
model = YOLO("../models/data_ver2_trained_model_11n.pt")

# Mở video
video_path = "../reality_test/video_splited/WIN_20250221_15_56_36_Pro_seg2.mp4"
cap = cv2.VideoCapture(video_path)

# Lấy thông tin video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Khởi tạo VideoWriter để lưu video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('infer/output_video_seg2.mp4', fourcc, fps, (frame_width, frame_height))

# Tạo folder để xuất các frame có conf > 0.7
export_folder = "export_frames2"
if not os.path.exists(export_folder):
    os.makedirs(export_folder)

frame_counter = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_counter += 1
    # Infer với mô hình
    results = model(frame)
    # Tạo một bản sao của frame để vẽ kết quả
    annotated_frame = frame.copy()

    # Lấy thông tin các bounding box từ kết quả infer
    boxes = results[0].boxes

    # Cờ để kiểm tra frame có dự đoán nào thỏa điều kiện conf > 0.7 không
    has_detection = False
    if boxes is not None and len(boxes) > 0:
        # Lấy tensor chứa tọa độ và độ chắc chắn
        bboxes = boxes.xyxy.cpu().numpy()  # numpy array với shape [N, 4]
        confs = boxes.conf.cpu().numpy()     # numpy array chứa conf với shape [N]
        # Duyệt qua từng dự đoán
        for bbox, conf in zip(bboxes, confs):
            if conf > 0.7:
                has_detection = True
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 128, 0), 2)
                label = f"{conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 0), 2)

    # Nếu có dự đoán với conf > 0.7, lưu frame vào folder
    if has_detection:
        output_path = os.path.join(export_folder, f"frame_{frame_counter:05d}.jpg")
        cv2.imwrite(output_path, annotated_frame)

    # Ghi frame đã annotate vào file video
    out.write(annotated_frame)
    cv2.imshow("YOLO Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
