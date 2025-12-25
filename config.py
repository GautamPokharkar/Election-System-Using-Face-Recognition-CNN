# config.py
import os

# MySQL credentials:
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'password',
    'database': 'election_db1'
}

# Email (SMTP) config - used to send OTPs.
SMTP_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'smtp_user': 'iramansari1595@gmail.com',    # your Gmail
    'smtp_password': 'aekdupgsniffsnhl'           # use app password for Gmail
}

# SYSTEM_EMAIL is no longer needed for multi-user registration
# SYSTEM_EMAIL = 'your.email@example.com'

# Paths
DATA_DIR = os.path.join(os.getcwd(), 'data')
FACES_DIR = os.path.join(DATA_DIR, 'faces')
MODEL_DIR = os.path.join(DATA_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'cnn_model.h5')

# capture settings
CAPTURE_COUNT = 50
