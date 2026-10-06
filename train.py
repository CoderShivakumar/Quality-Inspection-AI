from ultralytics import YOLO

# Load pretrained YOLO classification model
model = YOLO("yolo11n-cls.pt")

# Train
model.train(
    data=r"S:\2026-2027\new detection\Quality-Inspection-AI\dataset",
    epochs=50,
    imgsz=224,
    batch=4,
    patience=10,

    # Save training output to local C: drive
    project=r"C:\Users\sivak\yolo_training",
    name="defect_model",
    exist_ok=True,

    cache=False
)

print("Training completed!")