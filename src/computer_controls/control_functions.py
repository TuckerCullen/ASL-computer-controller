
import os
import datetime as dt
import webbrowser
import cv2
import time

# Keene's stuff
import platform, warnings
# Mac OS
if (platform.system() == "Darwin"):
    import osascript
# Windows
if (platform.system() == "Windows"):
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# Linux
if (platform.system() == "Linux"):
    from subprocess import call
# Input
from pynput.keyboard import Key
from pynput.mouse import Button

from pynput.keyboard import Controller as Key_Controller
from pynput.mouse import Controller as Mouse_Controller


def brightness(dir="up"):
    # TODO
    ...

def take_picture():
    """
    Opens webcam preview and takes a picture when you hit the spacebar.
    TODO: set this to run on timer or 1 sign to load preview and one sign to take picture
    """
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        cv2.imshow('Image Preview', rgb)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            out = cv2.imwrite('~/Desktop/capture.jpg', frame)
            break

    cap.release()
    cv2.destroyAllWindows()


def screenshot():
    """Takes a screenshot and saves to desktop"""

    time = str(dt.datetime.now().strftime("%Y-%m-%d_%h:%m"))
    os.system(f"screencapture -P ~/Desktop/screenshot{time}.jpeg")

def open_browser():
    """Opens google in default browser"""

    url = 'https://www.google.com/'
    webbrowser.open_new(url)

def check_weather():
    """
    Opens up google weather
    TODO: pretty sure this just goes to berkeley weather no matter where you are, add location based lookup
    """

    url = "https://www.google.com/search?q=weather&oq=weather&aqs=chrome..69i57j0i67l2j46i20i199i263i433i465i512j69i60l2j69i61j69i60.5279j1j7&sourceid=chrome&ie=UTF-8"
    webbrowser.open_new(url)
def search(query):
    url = "https://www.google.com/search?q=" + query
    webbrowser.open_new(url)
def volume(value):
    if (platform.system() == 'Linux'):
        # Linux
        call(["amixer", "-D", "pulse", "sset", "Master", str(value) + "%"])
    elif (platform.system() == 'Darwin'):
        # Mac OS
        osascript.osascript("set volume output volume " + str(value))
    elif (platform.system() == 'Windows'):
        # Windows
        dev = AudioUtilities.GetSpeakers()
        interface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMasterVolume(value, None)
    else:
        warnings.warn("ERROR: OS cannot be determined")
def volume_up(sens = 1):
    volume(1 * sens)
def volume_down(sens = 1):
    volume(-1 * sens)
def keypress(key, mode='tap'):
    """
    Key press control function
    modes =
        tap, immediately press and release
        hold, simulate a key press
        release, simulate a key release
    """
    keyboard = Key_Controller()

    if (mode == 'tap'):
        keyboard.press(key)
        keyboard.release(key)
    elif (mode == 'hold'):
        keyboard.press(key)
    elif (mode == 'release'):
        keyboard.release(key)

def mousepress(button, mode='tap'):
    """
    Mouse click control function
    modes =
        tap, immediately click and release
        hold, simulate a mouse click
        release, simulate a mouse release
        scroll, simulate scrolling. use button param as (dx, dy)
        double, double click mouse button, buttom param as (key, how many times)
    """
    mouse = Mouse_Controller()

    if (mode == 'tap'):
        mouse.press(button)
        mouse.release(button)
    elif (mode == 'hold'):
        mouse.press(button)
    elif (mode == 'release'):
        mouse.release(button)
    elif (mode == 'scroll'):
        mouse.scroll(button[0], button[1])
    elif (mode == 'double'):
        mouse.click(button[0], button[1])
# Open Application... (Set up with parameter)
if __name__ == "__main__":
    # take_picture()
    # screenshot()
    # open_browser()
    # check_weather()
    ...
