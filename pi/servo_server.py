from flask import Flask, request
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import time

app = Flask(__name__)
servo = Servo(18, pin_factory=PiGPIOFactory())

@app.route("/move", methods=["POST"])
def move_servo():
    position = request.json.get("pos")
    # if it's a number, set the position to the number
    if isinstance(position, (int, float)):
        if not -1.0 <= position <= 1.0:
            return "Invalid position value", 400
        servo.value = position
        return "OK", 200
    # if it's a string, set the position to the string
    elif isinstance(position, str):
        if position == "min":
            servo.min()
        elif position == "mid":
            servo.mid()
        elif position == "max":
            servo.max()
        else:
            return "Invalid position string", 400
    else:
        return "Invalid position type", 400
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# sudo pigpiod && python3 run.py