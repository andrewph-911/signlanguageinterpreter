import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = ["T", "B"]
# labels = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14",
#           "15","16","17","18","19","20","21","22","23","24","25","26","27",
#           "28","29"]

while True:
    success, img = cap.read()
    imgOutput1 = img.copy()
    imgOutput2 = img.copy()
    hands, img = detector.findHands(img)
    if hands:  # hand 1
        hand1 = hands[0]
        lmList = hand1['lmList']  # list of 21 Landmarks points
        x, y, w, h = hand1['bbox']  # Bounding Box info x,y,w,h

        imgWhite1 = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop1 = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape1 = imgCrop1.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize1 = cv2.resize(imgCrop1, (wCal, imgSize))
            imgResizeShape1 = imgResize1.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite1[:, wGap:wCal + wGap] = imgResize1
            prediction1, index = classifier.getPrediction(imgWhite1, draw=False)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize1 = cv2.resize(imgCrop1, (imgSize, hCal))
            imgResizeShape1 = imgResize1.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite1[hGap:hCal + hGap, :] = imgResize1
            prediction1, index = classifier.getPrediction(imgWhite1, draw=False)

        cv2.putText(img, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 4)
        cv2.rectangle(img, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 1)

        # print(labels[index], 'hand 1')
        cv2.imshow("ImageCrop", imgCrop1)
        cv2.imshow("ImageWhite", imgWhite1)

    if len(hands) == 2:  # hand 2
        hand2 = hands[1]  # list of 21 Landmarks points
        lmList = hand2['lmList']  # Bounding Box info x,y,w,h
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
            prediction2, index = classifier.getPrediction(imgWhite2, draw=False)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize2 = cv2.resize(imgCrop2, (imgSize, hCal))
            imgResizeShape2 = imgResize2.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite2[hGap:hCal + hGap, :] = imgResize2
            prediction2, index = classifier.getPrediction(imgWhite2, draw=False)

            cv2.putText(img, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 4)

            cv2.rectangle(img, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 1)

        cv2.imshow("ImageCrop2", imgCrop2)
        cv2.imshow("ImageWhite2", imgWhite2)

        def check_palindrome(my_str):
            if len(my_str) < 1:
                return True
            else:
                if my_str[0] == my_str[-1]:
                    return check_palindrome(my_str[1:-1])
                else:
                    return False
        if check_palindrome(labels[index]) == True:
            print(labels[index])
        else:
            print('false')

    cv2.imshow("Image", imgOutput1)
    cv2.imshow("Image", imgOutput2)
    # cv2.imshow("Image", img)
    cv2.waitKey(1)
