import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# Drawing a Line Beyween Each Detected Point
mpDraw = mp.solutions.drawing_utils

# Frame Rate
PrevTime = 0
CurrTime = 0

while True:
    success, img = cap.read()
    ImageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Results = hands.process(ImageRGB)
    print(Results.multi_hand_landmarks)
    
    if Results.multi_hand_landmarks:
        for HandLandmarks in Results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, HandLandmarks, mpHands.HAND_CONNECTIONS)
    
    # FPS (Frame Rate)
    CurrTime = time.time()
    FPS = 1/(CurrTime - PrevTime)
    PrevTime = CurrTime
    cv2.putText(img, str(int(FPS)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 250, 255), 2)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
