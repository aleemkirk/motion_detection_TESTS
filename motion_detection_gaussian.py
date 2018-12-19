#motion detection using adaptive Gaussian mixture model for background subtraction
import numpy as np 
import argparse
import imutils
from imutils.video import VideoStream
import cv2

video = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(history=60, detectShadows=False)
min_area = 2000

while True:
    frame = video.read()[1]
    grey = cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2GRAY) #turn image grey
    videoMasked = fgbg.apply(frame)
    #remove noise and display background subtracted and original video
    videoMasked = cv2.GaussianBlur(videoMasked, (3,3), 0)
    cv2.imshow('fgmask', frame)
    cv2.imshow('frame', videoMasked)

    #filtering image of low level pixel values and display resulting image1
    thresh = cv2.threshold(videoMasked, 100, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)) #kernel used in dilating process
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    cv2.imshow('threshold', thresh)

    #create Contours
    _,cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

    #loop over contours
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('fgmask', frame)
        
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
