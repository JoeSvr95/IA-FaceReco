from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# Contruyendo los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detection_method", type=str, default="hog",
    help="modelo de detección de rostro: 'hog' o 'cnn'")
args = vars(ap.parse_args())

# Cargar las caras reconocidas
print("[INFO] cargando encodings...")
data = pickle.loads(open("encodings.pickle", 'rb').read())

# Inicializando la camara
print("[INFO] iniciando camara...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    # Capturando los frames de la cámara
    frame = vs.read()

    # Convertir el frame de BGR a RGB y redimencionar para
    # que tenga anchura de 750 px
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    # Detectando las coordenadas (x, y) de las cajas delimitadoras
    # que corresponden a cada una de las caras del frame
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # Iterando sobre las caras
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Desconocido"

        # Verificando si encontramos un match
        if True in matches:
            # Encontrando todos los indices de las caras encontradas
            # e inicializar un diccionario que cuente el número total de matches
            matchedIndxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # Iterar sobre los indices que hicieron match y mantener un contador
            # por cada cara reconocida
            for i in matchedIndxs:
                name = data["names"][i].replace("-", " ").lower()
                counts[name] = counts.get(name, 0) + 1

            # Determinar la cara reconocida con la mayor cantidad de votos
            name = max(counts, key=counts.get)

        # Actualizando la lista de nombres
        names.append(name)

    # Iterando sobre las caras reconocidas
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # Re-escalar las coordenadas de la cara
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # Colocar el nombre sobre las caras
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            (0, 255, 0), 2)


    cv2.imshow("Frame", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
