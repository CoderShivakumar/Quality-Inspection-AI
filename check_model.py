from ultralytics import YOLO
import glob
import os

model = YOLO("runs/classify/train-2/weights/best.pt")

print("Classes:", model.names)

folders = [
    "dataset/train/good",
    "dataset/train/defect",
    "dataset/train/defects"
]

for folder in folders:

    images = glob.glob(os.path.join(folder, "*.*"))

    if not images:
        continue

    print("\n==============================")
    print("FOLDER:", folder)
    print("==============================")

    # Test up to 5 images from each class
    for image in images[:5]:

        result = model(image, verbose=False)[0]

        print("\nImage:", os.path.basename(image))

        for i, prob in enumerate(result.probs.data):
            print(
                f"{model.names[i]}: {float(prob):.4f}"
            )

        print(
            "Prediction:",
            model.names[result.probs.top1]
        )