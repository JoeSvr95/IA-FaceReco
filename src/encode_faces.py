from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# Directorio donde se ubica ESTE archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Parámetro para poder elegir entre 'hog' o 'cnn'
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
    help="ruta hacia el directorio de imágenes")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
    help="modelo de detección de caras a usar: 'hog' o 'cnn'")
args = vars(ap.parse_args())

# Recoger los paths de las imágenes
print("[INFO] quantificando caras...")
imagePaths = list(paths.list_images(args["dataset"]))

# Inicializando la lista de los encodings conocidos y nombres
knownEncodings = []
knownNames = []

# Iteranmos las rutas de las imágenes
for (i, imagePath) in enumerate(imagePaths):
    # Extraemos el nombre de la persona de la ruta
    print("[INFO] procesando imagen {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # Cargar la imagen input y convertira de BGR a dlib RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detectar las coordenadas (x, y) de los bounding boxes
    # que corresponden a cada una de las caras de las imágenes
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])

    # Computar las facciones de las caras
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Iterar los encodings
    for encoding in encodings:
        # Agregar cada encoding más el nombre a nuestro set
        # de caras conocidas y encodings
        knownEncodings.append(encoding)
        knownNames.append(name)

# Guardando los encodings de las caras más los nombres
print("[INFO] serializando los encodings...")
data = {"encodings": knownEncodings, "names": knownNames}

f = open("encodings.pickle", 'wb')
f.write(pickle.dumps(data))
f.close
