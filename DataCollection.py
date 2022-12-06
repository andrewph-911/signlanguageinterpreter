import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector: HandDetector = HandDetector(detectionCon=0.8, maxHands=2)

offset = 20
imgSize = 300

folder = "DataImages/Test"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:   # hand 1
        hand1 = hands[0]
        lmList = hand1['lmList']    # list of 21 Landmarks points
        x, y, w, h = hand1['bbox']  # Bounding Box info x,y,w,h

        imgWhite1 = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop1 = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape1 = imgCrop1.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize1 = cv2.resize(imgCrop1, (wCal, imgSize))
            imgResizeShap1 = imgResize1.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite1[:, wGap:wCal + wGap] = imgResize1

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize1 = cv2.resize(imgCrop1, (imgSize, hCal))
            imgResizeShape1 = imgResize1.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite1[hGap:hCal + hGap, :] = imgResize1

        cv2.imshow("ImageCrop1", imgCrop1)
        cv2.imshow("ImageWhite1", imgWhite1)

    if len(hands) == 2:     # hand 2
        hand2 = hands[1]    # list of 21 Landmarks points
        lmList = hand2['lmList']    # Bounding Box info x,y,w,h
        x, y, w, h = hand2['bbox']

        imgWhite2 = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop2 = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape2 = imgCrop2.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize2 = cv2.resize(imgCrop2, (wCal, imgSize))
            imgResizeShap2 = imgResize2.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite2[:, wGap:wCal + wGap] = imgResize2

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize2 = cv2.resize(imgCrop2, (imgSize, hCal))
            imgResizeShape2 = imgResize2.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite2[hGap:hCal + hGap, :] = imgResize2

        cv2.imshow("ImageCrop2", imgCrop2)
        cv2.imshow("ImageWhite2", imgWhite2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite1)
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite2)
        print(counter)
