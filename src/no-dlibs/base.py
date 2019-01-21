import numpy as numpy
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capturar frames
    ret, frame = cap.read()

    # Mostrar el frame resultante
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Cuando todo termine, liberar la captura
cap.release()
cv2.destroyAllWindows()