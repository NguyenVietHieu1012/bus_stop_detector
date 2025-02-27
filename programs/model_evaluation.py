from ultralytics import YOLO

# Tải mô hình đã huấn luyện
model = YOLO("C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/models/data_ver2_trained_model_11n.pt")

# Đánh giá mô hình trên tập kiểm tra
results = model.val(data="C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/dataset/YOLO_ver2/data.yaml", device=0, amp=False, batch=4, workers=0)

