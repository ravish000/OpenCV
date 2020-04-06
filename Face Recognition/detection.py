import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier("haarcascadeclassifier\haarcascade_face.xml");
cam=cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read('trainer//trainingData.yml')
id=0
font=cv2.FONT_HERSHEY_COMPLEX_SMALL

while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id, conf=rec.predict(gray[y:y+h,x:x+w])
        if(conf<50):
            if(id==1):
                id="Ravishankar"
            elif(id==2):
                id="hstendaredevil"
        else:
            id="Unknown"
        cv2.putText(img,str(id),(x,y+h),font,2,(0,0,0),2)
    cv2.imshow('Detection',img)
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()

        
