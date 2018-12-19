#src https://www.youtube.com/watch?v=NrkEXsCxkhw
#motion detection using a frame difference method
import numpy as np
import cv2 
import imutils
from imutils.video import VideoStream
import argparse
import time


#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=2000, help="minumum area size")
args = vars(ap.parse_args())

#if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    video = cv2.VideoCapture(0)

#Otherwise, we are reading from a video file
else:
    video = cv2.VideoCapture(args["video"])
    

frame1 = cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2GRAY)
frame2 = cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2GRAY)

#creating resizeable windows used for displaying output data
cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)
cv2.namedWindow("Difference", cv2.WINDOW_NORMAL)
cv2.namedWindow("Dilated", cv2.WINDOW_NORMAL)

#main loop
while True:
    #read and display image from webcam
    frame = video.read()[1]
    grey = cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2GRAY) #turn image grey
    cv2.imshow("webcam", frame)
    #finding the difference between two consecitive frames and displaying
    diff = cv2.absdiff(frame1, frame2)

    #displaying the absolute difference between two consecutive frames
    cv2.imshow("Difference", diff)

    #blurring, filtering and dilating all small changes using threshold
    blur = cv2.GaussianBlur(diff, (5,5), 0)
    thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)) #kernel used in dilating process
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    cv2.imshow("Dilated", dilated)

    #finding and displaying contours in dilated image
    img, cnts, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("webcam", frame)

    #reassigning new frames
    frame1 = frame2
    frame2 = grey



    #condition to exit loop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

video.release()
cv2.destoryallWindows()