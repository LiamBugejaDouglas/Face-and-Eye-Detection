import os
import cv2 as cv

def gen_negative_file():
    #Create file and saves negative images in a text file
    with open('neg.txt', 'w') as f:
        for filename in os.listdir('negative'):
            f.write('negative/'+ filename + '\n')

#Create file and saves positive images with position in a text file
#C:/Users/liamb/Desktop/MachineLearning/opencv/build/x64/vc15/bin/opencv_annotation.exe --annotations=pos.txt --images=positive/

#Save positive images in a vector file
#C:/Users/liamb/Desktop/MachineLearning/opencv/build/x64/vc15/bin/opencv_createsamples.exe -info posImg.txt 

#Train the model
#C:/Users/liamb/Desktop/MachineLearning/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data cascade/ -vec positive.vec -bg neg.txt -w 24 -h 24 -numPos 200 -numNeg 400 -numStages 13 
#-precalcValBufSize 6000 -precalcIdxBufsize 6000 -maxFalseAlarmRate 0.4 -minHitRate 0.999

def get_trained_model():

    cap = cv.VideoCapture(0)
    face_detect = cv.CascadeClassifier('cascade/cascade.xml')

    while True:

        ret, frame = cap.read()

        #Change image to greyscale.
        gray_scale= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #Return all the faces that are detected.
        faces = face_detect.detectMultiScale(gray_scale, 1.3, 5)
        #Draws a green rectangle around the face/s.
        for(x,y,w,h) in faces:
            cv.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 5)
        
        #Shows rectangles 
        cv.imshow('frame', frame)

        #Stop the program by pressing 'e'
        if(cv.waitKey(1) == ord('e')):
            break

    cap.release()
    cv.destroyAllWindows()