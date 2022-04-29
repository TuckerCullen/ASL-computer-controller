
import os
import datetime as dt
import webbrowser
import cv2
import time


def brightness(dir="up"):

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

def open_twitter():
    """Opens twitter in default browser"""

    url = 'https://twitter.com/home?lang=en'
    webbrowser.open_new(url)

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

def sleep():
    "makes the computer go to sleep"

    os.system("pmset sleepnow")


if __name__ == "__main__":

    # take_picture()
    # screenshot()
    # open_browser()
    # check_weather()
    # sleep()
    ...




