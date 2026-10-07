import av
import cv2
import numpy as np
import time


print("Python started")
print("Opening USB webcam using PyAV...")


# Try the USB webcam through Windows DirectShow
device = "video=C270 HD WEBCAM"
start = time.time()

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
        "Camera opened in",
        round(time.time() - start, 2),
        "seconds"
    )

except Exception as e:

    print("Could not open USB webcam")
    print("Error:")
    print(e)

    exit()


print("USB webcam opened successfully!")
print("Press Q to quit")


try:

    for frame in container.decode(video=0):

        # Convert PyAV frame to NumPy array
        image = frame.to_ndarray(format="bgr24")

        # Display using OpenCV
        cv2.imshow("USB Webcam - PyAV", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    container.close()
    cv2.destroyAllWindows()

    print("Camera stopped")