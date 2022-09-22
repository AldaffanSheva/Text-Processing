import cv2

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read('trainer/train.yml')
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

id = 4

names = ['', 'Anamika', 'Nitol', 'Liza', 'Bipul']

cam = cv2.VideoCapture(0)
cam.set(3, 640)  #width
cam.set(4, 480)  #height

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True :
    ret, img = cam.read()
    G = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        G,
        scaleFactor = 1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x,y,w,h) in faces :
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = faceRecognizer.predict(G[y:y + h, x:x + w])

        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("\n [INFO] Exit")
cam.release()
cv2.destroyAllWindows()