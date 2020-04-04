import os
import cv2
import numpy as np
from PIL import Image

recogniser = cv2.face.LBPHFaceRecognizer_create()
path = './dataset'


def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        print(ID)
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training", faceNp)
        cv2.waitKey(10)
    return IDs, faces


Ids, faces = getImagesWithID(path)
recogniser.train(faces, np.array(Ids))
recogniser.save('./recogniser/trainingData.yml')
cv2.destroyAllWindows()
