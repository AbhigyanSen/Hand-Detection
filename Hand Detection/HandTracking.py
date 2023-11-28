import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode = False, MaxHands = 2, DetectionConfidence = 0.5, TrackConfidence = 0.5):
        self.mode = mode
        self.MaxHands = MaxHands
        self.DetectionConfidence = DetectionConfidence
        self.TrackConfidence = TrackConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.MaxHands, self.DetectionConfidence, self.TrackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        
        def FindHands(self, img, draw = True):
            ImageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.Results = self.hands.process(ImageRGB)
            # print(Results.multi_hand_landmarks)

            if self.Results.multi_hand_landmarks:
                for HandLandmarks in self.Results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img, HandLandmarks, self.mpHands.HAND_CONNECTIONS)
            return img

        def FindPosition(self, img, HandNumber = 0, draw = True):
            LandmarkList = []
            if self.Results.multi_hand_landmarks:
                MyHand = self.Results.multi_hand_landmarks[HandNumber]
                
                for LandmarkId, Landmarks in enumerate (MyHand.Landmark):
                    print(LandmarkId, Landmarks)
                    Height, Width, Channel = img.shape
                    xCenter, yCenter = int(Landmarks.x * Width), int(Landmarks.y * Height)
                    print(LandmarkId, xCenter, yCenter)
                    LandmarkList.append([LandmarkId, xCenter, yCenter])
                    
                    if draw:
                        cv2.circle(img, (xCenter, yCenter), 15, (255, 0, 255), cv2.FILLED)
                        
            return LandmarkList
    
def main():
    PrevTime = 0
    CurrTime = 0
    cap = cv2.VideoCapture()
    
    Detector = HandDetector()
    
    while True:
        success, img = cap.read()
        img = Detector.FindHands(img)
        LandmarkList = Detector.FindPosition(img)
        if len(LandmarkList) != 0:
            print(LandmarkList[4])
                    
        CurrTime = time.time()
        FPS = 1 / (CurrTime - PrevTime)
        PrevTime = CurrTime
        cv2.putText(img, str(int(FPS)), (10, 78), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    
if __name__ == "__main__":
    main()
