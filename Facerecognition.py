import cv2, os, math, operator
from PIL import Image
from functools import reduce
face_cascade = cv2.CascadeClassifier("C:\\Users\\K556UQ\\PycharmProjects\\untitled2\\venv\\Lib\site-packages\\cv2\data\\haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)
while (cam.isOpened()):
    ret, img = cam.read()
    face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=(5, 5),flags=cv2.CASCADE_SCALE_IMAGE)
    recogname1 = "C:\\Users\\K556UQ\\Desktop\\python photo\\buildfacedata\\A1.jpg"
    recogname2 = "C:\\Users\\K556UQ\\Desktop\\python photo\\buildfacedata\\A2.jpg"
    loginname = cam.read()
    if ret == True :
        P1=Image.open(recogname1).histogram()
        P2=Image.open(recogname2).histogram()
        V1=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).histogram()
        diff = int(math.sqrt(reduce(operator.add, list(map(lambda a, b: (a - b) ** 2, P1,V1))) / len(P1)));print(diff);diff1=0
        diff1 = int(math.sqrt(reduce(operator.add, list(map(lambda a, b: (a - b) ** 2, P2, V1))) / len(P2)));print(diff1)
        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (128, 255, 0), 2)
            cv2.rectangle(img, (10, img.shape[0] - 20), (300, img.shape[0]), (0, 0, 0), -1)
            cv2.putText(img, "Find   " + str(len(face)) + "   face!", (10, img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
            if (diff<= 800):
                cv2.rectangle(img, (x, y+h), (x+200, y+h+30), (0,255,0), -1)
                cv2.putText(img, 'Wu Yan Lin ', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(img, 'Wu Yan Lin  '+str(diff), (160, img.shape[0] - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if (diff1<= 1000):
                cv2.rectangle(img, (x, y + h), (x + 200, y + h + 30), (0, 255, 0), -1)
                cv2.putText(img, 'Teacher', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(img, 'Teacher    ' + str(diff1), (160, img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
    cv2.imshow("Face recognition", img)
    k = cv2.waitKey(1)
    if k == ord("z") or k == ord("Z"):
        break
cam.release();cv2.destroyAllWindows()