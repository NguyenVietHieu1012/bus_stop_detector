from ultralytics import YOLO
import torch

# Chọn mô hình YOLO (có thể thay đổi thành phiên bản bạn muốn)
model = YOLO('../models/yolo11n.pt')

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")

results = model.train(
    data="C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/dataset/data.yaml",
    epochs=100,
    augment=True,  # Sử dụng tăng cường dữ liệu
    hsv_h=0.03,    # Điều chỉnh màu sắc
    hsv_s=0.7,     # Điều chỉnh độ bão hòa
    hsv_v=0.4,     # Điều chỉnh độ sáng
    scale=0.5,
    device=0,
    workers=0,
    amp=False,
    batch=4
)

# Lưu mô hình sau khi huấn luyện
model.save("C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/models/trained_model.pt")
