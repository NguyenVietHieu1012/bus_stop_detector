from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("../models/trained_model.pt")

# Define path to the image file
source = "../reality_test/20250210-1559-1651_frames/20250210-1559-1651_frames_imgs/FILE250210-155902-026259-M_seg1_frame_001468.png"

# Run inference on the source
results = model(source)  # list of Results objects