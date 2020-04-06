import cv2

#create a CascadeClassifier Object
face_cascade=cv2.CascadeClassifier("C:\\Users\\admin\\Desktop\\openCV\\OpenCV\\Faces.xml")

#reading an image 
image=cv2.imread("E:\\My document Ravish\\mmrs\\New folder\\Memories\\hstendaredevil.jpg")

#Reading an image as a grayscale image
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Search the coordinates of the image
faces=face_cascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=5)

print(type(faces))
print(faces)

for x,y,w,h in faces:
    image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)

resized=cv2.resize(image,(int(image.shape[1]),int(image.shape[0])))
cv2.imshow("gray",resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
