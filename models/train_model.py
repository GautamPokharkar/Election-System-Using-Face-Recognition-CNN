# models/train_model.py
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from config import MODEL_DIR, MODEL_PATH

def build_model(input_shape=(50,50,1), num_classes=2):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPool2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPool2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save():
    # Lazy import to avoid circular dependency
    from models.face_utils import load_faces_and_labels

    X, y = load_faces_and_labels()
    if len(X) == 0:
        raise Exception("No training data found. Register users first.")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(np.unique(y_enc))

    model = build_model(num_classes=num_classes)
    y_cat = to_categorical(y_enc, num_classes=num_classes)
    from tensorflow.keras.callbacks import EarlyStopping
    model.fit(X, y_cat, epochs=15, batch_size=16, validation_split=0.1, callbacks=[EarlyStopping(patience=3)])

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)

    # save label encoder
    import pickle
    with open(os.path.join(MODEL_DIR, 'le.pkl'), 'wb') as f:
        pickle.dump(le, f)

    return True
