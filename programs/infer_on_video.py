import cv2
import os
from ultralytics import YOLO

# Load mô hình YOLO
model = YOLO("../models/trained_model.pt")

# Mở video
video_path = "../tool/WIN_20250217_14_23_01_Pro_videos/WIN_20250217_14_23_01_Pro_seg4.mp4"
cap = cv2.VideoCapture(video_path)

# Lấy thông tin video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Khởi tạo VideoWriter để lưu video (với frame đã annotate)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('WIN_20250217_14_23_01_Pro_seg4_infer.mp4', fourcc, fps, (frame_width, frame_height))

# Tạo folder để xuất các frame có conf > 0.7 (không annotate)
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
    # Tạo một bản sao của frame để vẽ kết quả cho video
    annotated_frame = frame.copy()

    boxes = results[0].boxes

    has_detection = False
    if boxes is not None and len(boxes) > 0:
        bboxes = boxes.xyxy.cpu().numpy()  # numpy array với shape [N, 4]
        confs = boxes.conf.cpu().numpy()     # numpy array chứa conf với shape [N]
        for bbox, conf in zip(bboxes, confs):
            if conf > 0.7:
                has_detection = True
                # Vẽ bounding box trên frame copy dùng cho video
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Nếu có dự đoán với conf > 0.7, lưu frame gốc (không có annotation)
    if has_detection:
        output_path = os.path.join(export_folder, f"frame_{frame_counter:05d}.jpg")
        cv2.imwrite(output_path, frame)

    # Ghi frame đã annotate vào file video
    out.write(annotated_frame)
    cv2.imshow("YOLO Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
