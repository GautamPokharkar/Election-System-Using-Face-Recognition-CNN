# models/face_utils.py
import os
import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'faces')

def capture_faces_for_aadhaar(aadhaar, num_images=50):
    """
    Opens camera, captures `num_images` of the user's face, saves them in DATA_DIR/<aadhaar>/
    """
    user_dir = os.path.join(DATA_DIR, aadhaar)
    os.makedirs(user_dir, exist_ok=True)

    cam = cv2.VideoCapture(0)
    count = 0

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while count < num_images:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (50,50))
            cv2.imwrite(os.path.join(user_dir, f'{count}.png'), face_resized)
            count += 1
        cv2.imshow('Capture Faces - Press q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
    return count == num_images

def load_faces_and_labels():
    """
    Loads all faces from DATA_DIR and returns (X, y)
    X = numpy array of images
    y = corresponding aadhaar labels
    """
    X, y = [], []
    if not os.path.exists(DATA_DIR):
        return X, y

    for aadhaar in os.listdir(DATA_DIR):
        user_dir = os.path.join(DATA_DIR, aadhaar)
        if not os.path.isdir(user_dir):
            continue
        for file in os.listdir(user_dir):
            img_path = os.path.join(user_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = img.astype('float32') / 255.0
            X.append(img.reshape(50,50,1))
            y.append(aadhaar)
    X = np.array(X)
    y = np.array(y)
    return X, y
