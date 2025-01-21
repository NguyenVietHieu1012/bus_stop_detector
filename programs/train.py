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
    epochs=150,
    augment=True,
    device=0,
    workers=0,
    amp=False,
    batch=4,
    lr0=0.0442302,
    lrf=0.260472,
    momentum=0.658896,
    weight_decay=0.000496704,
    warmup_epochs=0.438574,
    box=0.0872528,
    hsv_h=0.0165629,
    cls=1.2936,
    translate=0.0941287
)

# Lưu mô hình sau khi huấn luyện
model.save("C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/models/trained_model_tuned.pt")
