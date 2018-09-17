import cv2
import numpy as np
import file
import json
from math import pi, atan2

# put -1/0/1 in VideoCapture()
cap = cv2.VideoCapture(0)
cv2.namedWindow('image')


def callback(x):
    pass

# read hsv settings from the settings file
def reading():
    hsv_settings = eval(open('file.txt', 'r').read())
    if hsv_settings is '':
        return {'ilowH' : 10, 'ihighH' : 43, 'ilowS' : 145, 'ihighS' : 255, 'ilowV' : 147, 'ihighV' : 255}
    else:
        return hsv_settings

# set HSV default values
hsv = reading()

# count the amount of balls
counter = 0

# create trackbars for color change
cv2.createTrackbar('lowH', 'image', hsv['ilowH'], 179, callback)
cv2.createTrackbar('highH', 'image', hsv['ihighH'], 179, callback)

cv2.createTrackbar('lowS', 'image', hsv['ilowS'], 255, callback)
cv2.createTrackbar('highS', 'image', hsv['ihighS'], 255, callback)

cv2.createTrackbar('lowV', 'image', hsv['ilowV'], 255, callback)
cv2.createTrackbar('highV', 'image', hsv['ihighV'], 255, callback)

# create trackbar for reset the HSV trackbars values
switch = '1 : Reset'
cv2.createTrackbar(switch, 'image', 0, 1, callback)

# create trackbar for chjange modes between angele and distance
mode_switch = '1-A/0-D'
cv2.createTrackbar(mode_switch, 'image', 0, 1, callback)


while True:
    ret, frame = cap.read()
    original = frame.copy()
    # grab the frame
    frame = original.copy()

    # decide if needed to reset the HSV trackbars parameters
    if cv2.getTrackbarPos(switch, 'image') == 1:
        cv2.setTrackbarPos('lowH', 'image', 0)
        cv2.setTrackbarPos('highH', 'image', 179)
        cv2.setTrackbarPos('lowS', 'image', 0)
        cv2.setTrackbarPos('highS', 'image', 255)
        cv2.setTrackbarPos('lowV', 'image', 0)
        cv2.setTrackbarPos('highV', 'image', 255)
        cv2.setTrackbarPos(switch, 'image', 0)

    # get trackbars position
    hsv['ilowH'] = cv2.getTrackbarPos('lowH', 'image')
    hsv['ihighH'] = cv2.getTrackbarPos('highH', 'image')
    hsv['ilowS'] = cv2.getTrackbarPos('lowS', 'image')
    hsv['ihighS'] = cv2.getTrackbarPos('highS', 'image')
    hsv['ilowV'] = cv2.getTrackbarPos('lowV', 'image')
    hsv['ihighV'] = cv2.getTrackbarPos('highV', 'image')

    # save HSV dictionary as a txt file
    with open('settings.txt', 'w') as settings:
        settings.write(json.dumps(hsv))
        settings.close()

    # get "mode" trackbar position
    mode = cv2.getTrackbarPos(mode_switch, 'image')

    # create a mask
    hsv_colors = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([hsv['ilowH'], hsv['ilowS'], hsv['ilowV']])
    higher_hsv = np.array([hsv['ihighH'], hsv['ihighS'], hsv['ihighV']])
    mask = cv2.inRange(hsv_colors, lower_hsv, higher_hsv)

    frame = cv2.bitwise_and(frame, frame, mask=mask)

    # create a cross kernel
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]],
                      dtype=np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    ret, mask = cv2.threshold(mask, 127, 255, 0)

    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours is not None:
        for cnt in contours:
            if len(cnt) > 50:

                x, y, w, h = cv2.boundingRect(cnt)
                ratio = w / h

                area_circle_from_rect = pi*((w/2)**2)

                (a, b), radius = cv2.minEnclosingCircle(cnt)
                center = (int(a), int(b))

                area_circle = pi*(radius ** 2)

                area_ratio = area_circle/area_circle_from_rect

                if 0.75 < ratio < 1.25 and 0.75 < area_ratio < 1.25 and radius > 5:
                    cv2.circle(original, center, int(radius), (255, 255, 0), 5)
                    # cv2.rectangle(original, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    counter += 1
                    (xtarget, y), _ = cv2.minEnclosingCircle(cnt)
                    xframe = frame.shape[1] / 2

                    # focal calculator
                    object_distance_from_camera = 10  # in cm
                    object_width = 12  # in cm
                    object_width_pixels = 60  # in pixels

                    # known focals
                    calc_focal = (object_width_pixels*object_distance_from_camera)/object_width
                    note8_focal = 538.5826771653543
                    yoga_focal = 540

                    # set which focal to use
                    f = yoga_focal

                    # find the angle and the  distance from the object
                    angle = atan2((xtarget - xframe), f) * (180/pi)
                    distance = (f*12.7)/(2*radius)

                    # choose what to display according to the trackbar data
                    if mode == 0:
                        data = distance
                    else:
                        data = angle

                    cv2.putText(original, str(int(data)), (int(x), int(y + 2 * radius)), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 0), 3)

    # show thresholded image
    cv2.putText(original, "Fuels: " + str(counter), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow('mask', frame)
    cv2.imshow('original', original)
    counter = 0
    k = cv2.waitKey(1) & 0xFF  # large wait time to remove freezing
    if k == 113 or k == 27:
        break