import face_recognition
import argparse
import pickle
import cv2

# Construyendo los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="ruta a la imagen a analizar")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
    help="modelo de detección de rostro a usar: 'hog' o 'cnn'")
args = vars(ap.parse_args())

# Cargar las caras y los encodigns
print("[INFO] cargando encodings...")
data = pickle.loads(open("encodings.pickle", 'rb').read())

# Cargar la imagen de input y convertirla a RGB
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detectar las coordenadas (x, y) de las cajas delimitadoras
# que corresponden a cada una de las caras de la imagen de input
print("[INFO] reconociendo caras...")
boxes = face_recognition.face_locations(rgb,
    model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

names = []

# Iterando sobre los encodings de las caras
for encoding in encodings:
    # Intentar hacer match de las caras de las imagen
    # de input con los encodings conocidos
    matches = face_recognition.compare_faces(data["encodings"],
	encoding)
    name = "Desconocido"

    # Chequear si tenemos matches
    if True in matches:
        # Encontrar todos los inidces de los matches e inicializar
        # diccionario para contar la cantidad de veces que una cara
        # hizo match
        matchedIndxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        # Iterar sobre los indeces de matches
        for i in matchedIndxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        # Determianr la cara reconocida con el mayor numero de votos
        name = max(counts, key=counts.get)

    names.append(name)

# Iterar sobre las caras reconocidas
for ((top, right, bottom, left), name) in zip(boxes, names):
    # Colocar el nombre de la persona identificada
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top -15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# Mostrar la imagen
cv2.imshow("Imagen", image)
cv2.waitKey(0)
