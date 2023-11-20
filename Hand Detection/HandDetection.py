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
            for LandmarkID, Landmark in enumerate(HandLandmarks.landmark):
                # print(LandmarkID, Landmark)
                # This print gives the co-ordinates in decimal values, hence converting them to pixel values
                Height, Width, Channel = img.shape
                
                # Co-ordinates for Center Position
                xCentre, yCentre = int(Landmark.x*Width), int(Landmark.y*Height)
                # print(LandmarkID, xCentre, yCentre)
                
                # Highlighting FingerTips
                if LandmarkID == 4:
                    cv2.circle(img, (xCentre, yCentre), 10, (255, 204, 153), cv2.FILLED)
                if LandmarkID == 8:
                    cv2.circle(img, (xCentre, yCentre), 10, (255, 204, 153), cv2.FILLED)
                if LandmarkID == 12:
                    cv2.circle(img, (xCentre, yCentre), 10, (255, 204, 153), cv2.FILLED)
                if LandmarkID == 16:
                    cv2.circle(img, (xCentre, yCentre), 10, (255, 204, 153), cv2.FILLED)
                if LandmarkID == 20:
                    cv2.circle(img, (xCentre, yCentre), 10, (255, 204, 153), cv2.FILLED)
                
            mpDraw.draw_landmarks(img, HandLandmarks, mpHands.HAND_CONNECTIONS)
    
    # FPS (Frame Rate)
    CurrTime = time.time()
    FPS = 1/(CurrTime - PrevTime)
    PrevTime = CurrTime
    cv2.putText(img, str(int(FPS)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 128), 2)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
