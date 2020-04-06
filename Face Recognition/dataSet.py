import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier("haarcascadeclassifier\haarcascade_face.xml");
cam=cv2.VideoCapture(0)

id=input("Enter user id : ")
sampleNum=0;

while(True):
    ret, img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1;
        cv2.imwrite("dataset/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100);
    cv2.imshow("datasetCreator",img);
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    elif sampleNum>20:
        break
cam.release()
cv2.destroyAllWindows()
