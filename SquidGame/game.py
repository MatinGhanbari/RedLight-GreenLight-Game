import threading

from cvzone.HandTrackingModule import HandDetector
# from numba import jit, coda
import numpy as np
import random
import time
import cv2
from pygame import mixer
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
drawing = mp.solutions.drawing_utils

mixer.init()

gameSensitivity = 6.5

WIDTH = 1280
HEIGHT = 720
# WIDTH = 640
# HEIGHT = 480

FPS = 45
vc = cv2.VideoCapture(0)
vc.set(3, WIDTH)
vc.set(4, HEIGHT)
vc.set(5, FPS)

detector = HandDetector(detectionCon=0.8, maxHands=2)


def playShootSound(sleepTime):
    mixer.music.load('resources/sounds/shoot.mp3')
    mixer.music.set_volume(1)
    mixer.music.play()
    time.sleep(sleepTime)
    mixer.music.stop()


def playGreenLightSound(sleepTime):
    mixer.music.load('resources/sounds/greenLight.mp3')
    mixer.music.set_volume(1)
    mixer.music.play()
    time.sleep(sleepTime)
    mixer.music.stop()


def playRedLightSound(sleepTime):
    mixer.music.load('resources/sounds/redLight.mp3')
    mixer.music.set_volume(1)
    mixer.music.play()
    time.sleep(sleepTime)
    mixer.music.stop()


def playGameSound(playTime):
    mixer.music.load('resources/sounds/gameSound.mp3')
    mixer.music.set_volume(1)
    mixer.music.play()
    time.sleep(playTime)
    mixer.music.stop()


# @jit(target="cuda")
def start():
    isRunning = [True]
    FFrame = None
    var = [False]
    sTime = time.time()
    rnd = 0  # random.Random().randint(0, 4)
    testVar = [False]

    while True:
        _, firstFrame = vc.read()
        firstFrame = cv2.flip(firstFrame, 1)
        hands, firstFrame = detector.findHands(firstFrame, flipType=False, draw=True)
        cv2.putText(img=firstFrame, text="Press Space For Start Game", org=(100, 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(150, 0, 255), thickness=6)
        cv2.putText(img=firstFrame, text="Game starts in 2 seconds", org=(115, 150),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(150, 0, 255), thickness=6)
        if len(hands) >= 2:
            break
        cv2.namedWindow('capture', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('capture', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('capture', firstFrame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    time.sleep(2)
    z = 0
    startTime = time.time()
    threading.Thread(target=playGreenLightSound, args=())
    while isRunning[0]:

        _, frame = vc.read()
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        # landFrame = frame.copy()
        if res.pose_landmarks:
            drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if res.pose_landmarks.landmark[11].visibility > 0.7 and res.pose_landmarks.landmark[12].visibility > 0.7:
                z = res.pose_landmarks.landmark[11].x - res.pose_landmarks.landmark[12].x
                z *= 4
                z = round(z, 2)
            else:
                z = 0
            if z < 0: z = 0

        print("#Z: ", z)
        if FFrame is None:
            FFrame = frame.copy()

        # if res.pose_landmarks:
        #     if res.pose_landmarks.landmark[12].visibility > 0.7 and res.pose_landmarks.landmark[31].visibility > 0.7 and \
        #             res.pose_landmarks.landmark[5].visibility > 0.7:
        #         frameTemp = frame[round(res.pose_landmarks.landmark[5].y * frame.shape[0] - 50):
        #                           round(res.pose_landmarks.landmark[31].y * frame.shape[0] + 50),
        #                     round(res.pose_landmarks.landmark[12].x * frame.shape[1] - 50)
        #                     :round(res.pose_landmarks.landmark[31].x * frame.shape[1] + 50), :]
        #         FFrameTemp = FFrame[round(res.pose_landmarks.landmark[5].y * frame.shape[0] - 50):
        #                             round(res.pose_landmarks.landmark[31].y * frame.shape[0] + 50),
        #                      round(res.pose_landmarks.landmark[12].x * frame.shape[1] - 50)
        #                      :round(res.pose_landmarks.landmark[31].x * frame.shape[1] + 50), :]

        minX, minY = 0, 0
        w, h = frame.shape[1], frame.shape[0]
        maxX, maxY = frame.shape[1], frame.shape[0]
        if res.pose_landmarks:
            for i in range(len(res.pose_landmarks.landmark)):
                if i >= 11 and res.pose_landmarks.landmark[i].visibility > 0.8:
                    if res.pose_landmarks.landmark[i].x * w > maxX:
                        maxX = res.pose_landmarks.landmark[i].x * w
                    if res.pose_landmarks.landmark[i].y * h > maxY:
                        maxY = res.pose_landmarks.landmark[i].y * h
                    if res.pose_landmarks.landmark[i].x * w < minX:
                        minX = res.pose_landmarks.landmark[i].x * w
                    if res.pose_landmarks.landmark[i].y * h < minY:
                        minY = res.pose_landmarks.landmark[i].y * h
        maxX, maxY = round(maxX + 0), round(maxY + 0)
        minX, minY = round(minX - 0), round(minY - 0)
        frameTemp = frame[minY:maxY, minX:maxX, :]
        FFrameTemp = FFrame[minY:maxY, minX:maxX, :]

        try:
            diff = cv2.absdiff(FFrameTemp, frameTemp)
            diff = cv2.GaussianBlur(diff, (5, 5), 10)
        except Exception as e:
            diff = cv2.absdiff(FFrame, frame)
            diff = cv2.GaussianBlur(diff, (5, 5), 10)
        diff2 = diff
        diff2 = cv2.resize(diff2, (0, 0), fx=0.3, fy=0.3)

        img = frame.copy()
        img[(img.shape[0] - diff2.shape[0]):img.shape[0], (img.shape[1] - diff2.shape[1]):img.shape[1]] = diff2

        color = (0, 0, 255)
        text = "Dont Move"
        if var[0]:
            color = (0, 255, 0)
        text = "You Can Move"

        cv2.putText(img=img, text=text, org=(10, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=color, thickness=4)

        cv2.namedWindow('capture', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('capture', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('capture', img)

        print("#Diff: ", np.mean(diff))

        # if (var[0] is False) and np.mean(diff) > (12.5 + 100 * z):
        if (var[0] is False) and np.mean(diff) > gameSensitivity:
            threading.Thread(target=playShootSound, args=(2,)).start()
            print('You are dead! ❌')
            break

        if cv2.waitKey(1) & 0xFF == ord(' '):
            testVar[0] = True
            print('You are win! ✔')
            break

        if var[0]:
            eTime = time.time()
            if eTime - sTime >= rnd:
                playRedLightSound(0.25)
                rnd = 3
                sTime = eTime
                var[0] = False
            FFrame = frame.copy()
        else:
            eTime = time.time()
            if eTime - sTime >= rnd:
                playGreenLightSound(0.5)
                rnd = random.random() * 4.5 + 1
                sTime = eTime
                var[0] = True
                threading.Thread(target=playGameSound, args=(rnd,)).start()

    if testVar[0]:
        cv2.putText(img=FFrame, text="You Win!", org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.3, color=(0, 255, 0), thickness=6)
    else:
        cv2.putText(img=FFrame, text="You Dead!", org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.3, color=(0, 0, 255), thickness=6)
    cv2.imshow('capture', FFrame)
    cv2.waitKey(0)
    isRunning[0] = False
    cv2.destroyAllWindows()
    print("Play Time: ", round(time.time() - startTime), "s")


if __name__ == '__main__':
    start()
