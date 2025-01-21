from ultralytics import YOLO
from ray import tune

model = YOLO('../models/trained_model.pt')
model_tune = model.tune(
    data='C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/dataset/data.yaml',
    use_ray=True,
    epochs=15,
    space={
        "lr0": tune.uniform(1e-5, 1e-1),
        "lrf": tune.uniform(0.01, 1.0),
        "momentum": tune.uniform(0.6, 0.98),
        "weight_decay": tune.uniform(0.0, 0.001),
        "warmup_epochs": tune.uniform(0.0, 5.0),
        "box": tune.uniform(0.02, 0.2),
        "hsv_h": tune.uniform(0.0, 0.1),
        "cls": tune.uniform(0.2, 4.0),
        "translate": tune.uniform(0., 0.9)
    },
    device=0, amp=False, batch=4, workers=0
)
