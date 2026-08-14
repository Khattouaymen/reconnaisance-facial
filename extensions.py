"""
Extensions Flask — initialisées ici pour éviter les imports circulaires.

Pourquoi ce fichier ?
- app.py a besoin de db et login_manager
- models.py a aussi besoin de db
- Si les deux importent depuis le même fichier → import circulaire !
- Solution : on crée les extensions ici, et les deux fichiers importent depuis extensions.py
"""
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

# --- Base de données ---
db = SQLAlchemy()

# --- Gestionnaire de connexion ---
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.login_message = 'Veuillez vous connecter pour accéder à cette page.'
