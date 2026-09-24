import serial
import time

esp32 = serial.Serial("COM9", 115200, timeout=1)

time.sleep(2)

print("Connected to ESP32!")

esp32.write(b"GOOD\n")
print("Sent: GOOD")

time.sleep(2)

esp32.write(b"DEFECT\n")
print("Sent: DEFECT")

esp32.close()

print("Connection closed.")