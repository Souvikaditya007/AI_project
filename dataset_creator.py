import cv2
faceDetect=cv2.CascadeClassifier('E:\\python\\python_projects\\Computer_Vision_A_Z_Template_Folder\\Module 1 - Face Recognition\\haarcascade_frontalface_default.xml')

cam=cv2.VideoCapture(0)
id=input('Enter user name:')

sampleNum=0
while(True):
    ret, img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        sampleNum+=1
        cv2.imwrite('./dataset/User.'+str(id)+'.'+str(sampleNum)+'.jpg', gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),2)
        cv2.waitKey(100)
    cv2.imshow('Face',img)
    cv2.waitKey(1)
    if(sampleNum>20):
        break
cam.release()
cv2.destroyAllWindows()
