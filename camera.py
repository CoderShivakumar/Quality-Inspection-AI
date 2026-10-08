from ultralytics import YOLO
import av
import cv2
import serial
from collections import Counter
import time


# ==========================================
# 1. LOAD YOLO MODEL
# ==========================================

MODEL_PATH = r"runs/classify/train-2/weights/best.pt"

model = YOLO(MODEL_PATH)

print("Model loaded!")
print("Classes:", model.names)


# ==========================================
# 2. ESP32 SETTINGS
# ==========================================

ESP32_PORT = "COM9"       # CHANGE THIS TO YOUR ESP32 COM PORT
BAUD_RATE = 115200


# ==========================================
# 3. CONNECT ESP32
# ==========================================

try:

    esp32 = serial.Serial(
        ESP32_PORT,
        BAUD_RATE,
        timeout=1
    )

    time.sleep(2)

    esp32.reset_input_buffer()

    print("ESP32 connected!")

except Exception as e:

    print("ESP32 connection failed!")
    print("Error:", e)

    exit()


# ==========================================
# 4. OPEN C270 USING PYAV
# ==========================================

print("Opening C270 HD WEBCAM...")

device = "video=Logi C270 HD WebCam"

start_time = time.time()

try:

    container = av.open(
        device,
        format="dshow",
        options={
            "video_size": "640x480",
            "framerate": "30"
        }
    )

    print(
        "Camera opened in:",
        round(time.time() - start_time, 2),
        "seconds"
    )

except Exception as e:

    print("C270 HD WEBCAM could not be opened!")
    print("Error:", e)

    esp32.close()

    exit()


print("C270 HD WEBCAM started!")

print("--------------------------------")
print("QUALITY INSPECTION SYSTEM")
print("--------------------------------")
print("Show the bearing")
print("Press Q to quit")
print("--------------------------------")


# ==========================================
# 5. SETTINGS
# ==========================================

TOTAL_FRAMES = 10

MIN_CONFIDENCE = 0.70

prediction_list = []

final_prediction = "WAITING"

# IMPORTANT:
# Prevent sending the same command repeatedly
last_sent_prediction = ""


# ==========================================
# 6. CAMERA + YOLO LOOP
# ==========================================

try:

    for frame in container.decode(video=0):

        # ----------------------------------
        # Convert PyAV frame to OpenCV image
        # ----------------------------------

        image = frame.to_ndarray(format="bgr24")


        # ----------------------------------
        # YOLO prediction
        # ----------------------------------

        results = model(
            image,
            verbose=False
        )

        result = results[0]

        probabilities = result.probs.data


        # ----------------------------------
        # CLASS PROBABILITIES
        #
        # 0 = defect
        # 1 = good
        # ----------------------------------

        defect_probability = float(
            probabilities[0]
        )

        good_probability = float(
            probabilities[1]
        )


        # ----------------------------------
        # CURRENT PREDICTION
        # ----------------------------------

        if good_probability > defect_probability:

            current_prediction = "GOOD"

            confidence = good_probability

        else:

            current_prediction = "DEFECT"

            confidence = defect_probability


        # ----------------------------------
        # STORE ONLY CONFIDENT RESULTS
        # ----------------------------------

        if confidence >= MIN_CONFIDENCE:

            prediction_list.append(
                current_prediction
            )


        # ==================================
        # 7. 10-FRAME MAJORITY DECISION
        # ==================================

        if len(prediction_list) >= TOTAL_FRAMES:

            counts = Counter(prediction_list)

            final_prediction = counts.most_common(1)[0][0]


            print("--------------------------------")
            print("10-frame decision")
            print("GOOD   :", counts["GOOD"])
            print("DEFECT :", counts["DEFECT"])
            print("FINAL  :", final_prediction)


            # ==================================
            # 8. SEND COMMAND ONLY IF CHANGED
            # ==================================

            if final_prediction != last_sent_prediction:

                # Clear old ESP32 data
                esp32.reset_input_buffer()


                # ----------------------------------
                # SEND GOOD
                # ----------------------------------

                if final_prediction == "GOOD":

                    esp32.write(
                        b"GOOD\n"
                    )

                    print(
                        "Sent to ESP32: GOOD"
                    )


                # ----------------------------------
                # SEND DEFECT
                # ----------------------------------

                elif final_prediction == "DEFECT":

                    esp32.write(
                        b"DEFECT\n"
                    )

                    print(
                        "Sent to ESP32: DEFECT"
                    )


                # Remember what was sent
                last_sent_prediction = final_prediction


                # ==================================
                # 9. WAIT FOR ESP32 RESPONSE
                # ==================================

                start_wait = time.time()

                response_received = False

                while time.time() - start_wait < 2:

                    if esp32.in_waiting > 0:

                        response = (
                            esp32.readline()
                            .decode(errors="ignore")
                            .strip()
                        )

                        if response:

                            print(
                                "ESP32:",
                                response
                            )

                            if final_prediction in response:

                                response_received = True

                                break

                    time.sleep(0.01)


                # ----------------------------------
                # CHECK RESPONSE
                # ----------------------------------

                if not response_received:

                    print(
                        "ESP32: No acknowledgment received"
                    )

            else:

                print(
                    "Same result - command not sent again"
                )


            # ----------------------------------
            # CLEAR PREDICTIONS
            # ----------------------------------

            prediction_list.clear()


        # ==================================
        # 10. DISPLAY CURRENT PREDICTION
        # ==================================

        cv2.putText(
            image,
            f"CURRENT: {current_prediction}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )


        cv2.putText(
            image,
            f"CONFIDENCE: {confidence:.2f}",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        cv2.putText(
            image,
            f"FINAL: {final_prediction}",
            (30, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3
        )


        cv2.putText(
            image,
            f"Samples: {len(prediction_list)}/{TOTAL_FRAMES}",
            (30, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ==================================
        # 11. SHOW CAMERA
        # ==================================

        cv2.imshow(
            "C270 Bearing Quality Inspection",
            image
        )


        # ==================================
        # 12. QUIT
        # ==================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ==========================================
# 13. CLEANUP
# ==========================================

finally:

    container.close()

    cv2.destroyAllWindows()

    esp32.close()

    print("--------------------------------")
    print("Camera stopped")
    print("ESP32 disconnected")
    print("--------------------------------")