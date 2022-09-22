import cv2
import numpy as np
import os
from PIL import Image

dir = r"E:\Job\Zettabyte\Face-Recognition-Haar-Cascade-master\Face-Recognition-Haar-Cascade-master\dataset"

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImageLabel(dir) :
  imagePaths = [os.path.join(dir,f) for f in os.listdir(dir)]
  facesamples = []
  faceID = []
  for imagePath in imagePaths:
    PILImg = Image.open(imagePath).convert('L')
    imgNum = np.array(PILImg, 'uint8')
    faceID = int(os.path.split(imagePath)[-1].split(".")[1])
    faces = detector.detectMultiScale(imgNum)
    for (x,y,w,h) in faces:
      facesamples.append(imgNum[y:y+h,x:x+w])
      faceID.append(faceID)

    return facesamples, faceID


print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, faceID = getImageLabel(dir)
faceRecognizer.train(faces, np.array(faceID))

# Save the model into trainer/trainer.yml
faceRecognizer.write('trainer/train.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(faceID))))