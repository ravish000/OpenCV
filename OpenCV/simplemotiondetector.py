#Motion Detection

import cv2,time

first_frame=None

#Create a VideoCapture object to record video
video=cv2.VideoCapture(0) 

while True:
    chek, frame=video.read()

    #Convert the frame color to gray scale
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Convert the gray scale frame to Gaussian Blur
    gray=cv2.GaussianBlur(gray,(21,21),0)

    #This is use to store the first image/frame of the Video
    if first_frame is None:
        first_frame=gray
        continue

    #Calculates the difference between the fisrt frame and other frames
    delta_frame=cv2.absdiff(first_frame,gray)

    #Provide a theshold value
    thresh_delta=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    thresh_delta=cv2.dilate(thresh_delta, None, iterations=0 )

    #Define the contour area.Basically,add the borders.
    (_,cnts,_)=cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #Removes noise and shadow
    for contour in cnts:
        if cv2.contourArea(contour)<1000:
            continue

        #Create Recrangular box around the object in the frame
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

        cv2.imshow('frame',frame)
        cv2.imshow('Capturing',gray)
        cv2.imshow('delta',delta_frame)
        cv2.imshow('thresh',thresh_delta)
        key=cv2.waitKey(1)

        if key==ord('q'):
            break;

video.release()
cv2.destroyAllWindows()
