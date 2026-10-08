import serial
import time

arduino = serial.Serial("COM15", 115200, timeout=2)

time.sleep(2)

print("Sending GOOD")

arduino.write(b"GOOD\n")
arduino.flush()

time.sleep(2)

while arduino.in_waiting:
    print("Arduino:", arduino.readline().decode(errors="ignore").strip())

arduino.close()