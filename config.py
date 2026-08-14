"""
Configuration du Système de Présence — SAMHA METAL S.A.R.L
"""
import os


class Config:
    """Classe de configuration principale"""

    # --- Clé secrète pour les sessions Flask ---
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'samha-metal-secret-key-2025'

    # --- Base de données SQLite ---
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'data', 'database.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- Dossier pour les photos des employés ---
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'faces')

    # --- Reconnaissance faciale ---
    FACE_SIMILARITY_THRESHOLD = 0.5    # Seuil minimum pour identifier un employé
    POINTAGE_COOLDOWN = 30             # Secondes entre 2 pointages du même employé

    # --- Horaires de travail ---
    WORK_START_TIME = '08:00'          # Heure de début (pour calcul des retards)
    WORK_END_TIME = '17:00'            # Heure de fin

    # --- Compte admin par défaut ---
    DEFAULT_ADMIN_USER = 'admin'
    DEFAULT_ADMIN_PASS = 'samha2025'   # À changer après le premier login
