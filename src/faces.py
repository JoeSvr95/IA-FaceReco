import numpy as numpy
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {} # Diccionario de labels
with open("labels.pickle", 'rb') as f: # Leemos los lables
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()} # Invertimos el diccionario
cap = cv2.VideoCapture(0)

while(True):
    # Capturar frames
    ret, frame = cap.read()
    
    # Convertimos el frame a escala de grises
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_scale, scaleFactor=1.5, minNeighbors=5)
    
    # Iteración de las coordenadas de la cara
    for (x, y, w, h) in faces:
        roi_gray = gray_scale[y:y+h, x:x+w] # Región de interés en la escala de grises

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        color = (255, 0, 0) #BGR
        stroke = 2
        width = x + w
        height = y + h
        cv2.rectangle(frame, (x, y), (width, height), color, stroke)

    # Mostrar el frame resultante
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Cuando todo termine, liberar la captura
cap.release()
cv2.destroyAllWindows()