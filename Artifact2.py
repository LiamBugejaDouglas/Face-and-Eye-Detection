import cv2 as cv
import numpy as np
import os

#Video capture.
cap = cv.VideoCapture(0)

#Pre-trained face detection.
face_detect = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
#Pre-trained eye detection.
eye_detect = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

while True:

    ret, frame = cap.read()

    #Change image to greyscale.
    gray_scale= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #Return all the faces that are detected.
    faces = face_detect.detectMultiScale(gray_scale, 1.3, 5)
    #Draws a green rectangle around the face/s.
    for(x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 5)
        
        #Search through the location of the face and find the eyes
        eye_gray_scale = gray_scale[y:y+w, x:x+w]
        eye_color_scale = frame[y:y+h, x:x+w]

        #Return all of the eyes that are detected
        eyes = eye_detect.detectMultiScale(eye_gray_scale,1.3,5)
        #Draw a red rectangle around the eye/s
        for(ex,ey,ew,eh) in eyes:
            cv.rectangle(eye_color_scale, (ex,ey), (ex+ew, ey+eh), (255,0,0),5)

    #Shows rectangles 
    cv.imshow('frame', frame)

    #Stop the program by pressing 'e'
    if(cv.waitKey(1) == ord('e')):
        break

cap.release()
cv.destroyAllWindows()