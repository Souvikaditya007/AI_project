import cv2

faceDetect = cv2.CascadeClassifier(
    'E:\\python\\python_projects\\Computer_Vision_A_Z_Template_Folder\\Module 1 - Face Recognition\\haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read(
    'E:\\python\\python_projects\\Computer_Vision_A_Z_Template_Folder\\Module 1 - Face Recognition\\recogniser\\trainingData.yml')
id = 0
while (True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        id, conf = rec.predict(gray[y:y + h, x:x + w])
        cv2.putText(img, str(id), (x, x), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        print(id)
        print(conf)
    cv2.imshow("Face", img)
    if (cv2.waitKey(1) == ord('q')):
        break
cam.release()
cv2.destroyAllWindows()
