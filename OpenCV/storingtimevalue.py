import cv2, time, pandas
from datetime import datetime

first_frame=None
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["start","End"])

video=cv2.VideoCapture(0)

while True:
    check,frame=video.read()
    status=0
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

    for contour in cnts:
        if cv2.contourArea(contour)<1000:
            continue
        status=1
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.imshow('frame',frame)
        cv2.imshow('Capturing',gray)
        cv2.imshow('delta',delta_frame)
        cv2.imshow('thresh',thresh_delta)
        key=cv2.waitKey(1)

        if key==ord('q'):
            break;

    status_list.append(status)

    status_list=status_list[-2:]

    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())
print(status_list)
print(time)

for i in range(0,len(times),2):
    df=df.append({"start":times[i],"End":times[i+1]},ignore_index=True)
    df.to_csv("C:\\Users\\admin\\Desktop\\openCV\\OpenCV\\Times.csv")

video.release()
cv2.destroyAllWindows()
