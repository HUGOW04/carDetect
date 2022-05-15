from tkinter import Frame
import cv2


object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=170)

cap = cv2.VideoCapture("road.mp4")

while True:
    car = 0
    ret, frame = cap.read()
    height, width, _ = frame.shape
    roi = frame[300:500,450:850]
    mask = object_detector.apply(roi)
    constours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in constours:
        area = cv2.contourArea(cnt)
        if area < 100:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(roi,(x,y),(x + w, y + h), (0,255,0),3)

    cv2.imshow("roi",roi,)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask",mask)
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
