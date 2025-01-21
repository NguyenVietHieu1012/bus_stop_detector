from ultralytics import YOLO

# Tải mô hình đã huấn luyện
model = YOLO("C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/models/trained_model_tuned.pt")

# Đánh giá mô hình trên tập kiểm tra
results = model.val(data="C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/dataset/data.yaml", device=0, amp=False, batch=4, workers=0)

# In các kết quả đánh giá
print(f"Precision: {results.box.map:.2f}")
print(f"Recall: {results.box.map50:.2f}")
print(f"mAP@0.5: {results.box.map50:.2f}")
print(f"mAP@0.5:0.95: {results.box.map:.2f}")