from ultralytics import YOLO
import cv2
from collections import Counter

# ==========================================
# LOAD MODEL
# ==========================================

model = YOLO("runs/classify/train-2/weights/best.pt")

print("Model loaded!")
print("Classes:", model.names)

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Camera could not be opened")
    exit()

print("Camera started")
print("Show the bearing")
print("Press Q to quit")

# ==========================================
# SETTINGS
# ==========================================

TOTAL_FRAMES = 10
MIN_CONFIDENCE = 0.70

prediction_list = []

final_prediction = "WAITING"

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera")
        break

    # YOLO prediction
    results = model(frame, verbose=False)
    result = results[0]

    probabilities = result.probs.data

    defect_probability = float(probabilities[0])
    good_probability = float(probabilities[1])

    # Highest probability
    if good_probability > defect_probability:
        current_prediction = "GOOD"
        confidence = good_probability
    else:
        current_prediction = "DEFECT"
        confidence = defect_probability

    # --------------------------------------
    # Collect only confident predictions
    # --------------------------------------

    if confidence >= MIN_CONFIDENCE:

        prediction_list.append(current_prediction)

    # --------------------------------------
    # Once 10 predictions are collected
    # --------------------------------------

    if len(prediction_list) >= TOTAL_FRAMES:

        counts = Counter(prediction_list)

        final_prediction = counts.most_common(1)[0][0]

        print("--------------------------------")
        print("10-frame decision:")
        print("GOOD   :", counts["GOOD"])
        print("DEFECT :", counts["DEFECT"])
        print("FINAL  :", final_prediction)

        # Start next decision cycle
        prediction_list.clear()

    # ======================================
    # DISPLAY
    # ======================================

    cv2.putText(
        frame,
        f"CURRENT: {current_prediction}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"CONFIDENCE: {confidence:.2f}",
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FINAL: {final_prediction}",
        (30, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"Samples: {len(prediction_list)}/{TOTAL_FRAMES}",
        (30, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Bearing Quality Inspection", frame)

    # ======================================
    # QUIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Camera stopped")