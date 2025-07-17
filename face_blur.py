import cv2 as cv
import numpy as np

#initialize webcam
capture = cv.VideoCapture(0)

#load face cascade
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    #read frame from webcam
    ret, frame = capture.read()
    #flip frame horizontally for mirror effect
    frame = cv.flip(frame, 1)

    if not ret:
        break
    
    #convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    #draw rectangle around faces
    for (x, y, w, h) in faces:
        #cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_roi = frame[y:y+h, x:x+w]
        ksize = int(min(w, h) / 2) | 1 
        blurred_face = cv.GaussianBlur(face_roi, (ksize, ksize), 0)

        frame[y:y+h, x:x+w] = blurred_face
    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()