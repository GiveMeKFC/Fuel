import cv2
import numpy as np
from math import pi

cap = cv2.VideoCapture(1)
cv2.namedWindow('image')


def callback(x):
    pass


ilowH = 10
ihighH = 43
ilowS = 65
ihighS = 255
ilowV = 189
ihighV = 255

# create trackbars for color change
cv2.createTrackbar('lowH', 'image', ilowH, 179, callback)
cv2.createTrackbar('highH', 'image', ihighH, 179, callback)

cv2.createTrackbar('lowS', 'image', ilowS, 255, callback)
cv2.createTrackbar('highS', 'image', ihighS, 255, callback)

cv2.createTrackbar('lowV', 'image', ilowV, 255, callback)
cv2.createTrackbar('highV', 'image', ihighV, 255, callback)

while (True):
    ret, frame = cap.read()
    original = frame.copy()
    # grab the frame
    frame = original.copy()

    # get trackbar positions
    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')
    ilowV = cv2.getTrackbarPos('lowV', 'image')
    ihighV = cv2.getTrackbarPos('highV', 'image')

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

    frame = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]],
                      dtype=np.uint8)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    ret, black = cv2.threshold(mask, 127, 255, 0)

    im2, contours, hierarchy = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours is not None and len(contours) > 85:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / h

            area_circle_from_rect = pi*((w/2)**2)

            (a, b), radius = cv2.minEnclosingCircle(cnt)
            center = (int(a), int(b))
            radius = int(radius)

            area_circle = pi*(radius ** 2)

            area_ratio = area_circle/area_circle_from_rect

            if 0.75 < ratio < 1.25 and 0.75 < area_ratio < 1.25 and area_circle_from_rect > 2500:
                cv2.circle(original, center, radius, (255, 255, 0), 6)
                #cv2.rectangle(original, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.putText(original, "Fuel", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0),2)


    # show thresholded image
    cv2.imshow('mask', frame)
    cv2.imshow('original', original)
    k = cv2.waitKey(1) & 0xFF  # large wait time to remove freezing
    if k == 113 or k == 27:
        break