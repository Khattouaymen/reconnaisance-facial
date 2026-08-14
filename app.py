"""
Application Flask — Système de Présence SAMHA METAL S.A.R.L

Ce fichier est le point d'entrée de l'application.
Il initialise Flask, connecte les extensions, définit les routes,
et crée la base de données au premier lancement.
"""
from flask import Flask, render_template
from config import Config
from extensions import db, login_manager
from datetime import date, datetime
import os


# ============================================================
# INITIALISATION DE L'APPLICATION
# ============================================================

app = Flask(__name__)
app.config.from_object(Config)

# Connecter les extensions à l'application
db.init_app(app)
login_manager.init_app(app)

# Créer les dossiers nécessaires (data/ et data/faces/)
os.makedirs(app.config.get('UPLOAD_FOLDER', 'data/faces'), exist_ok=True)

# Importer les modèles APRÈS l'initialisation de db (important !)
from models import Employee, Attendance, User
from werkzeug.security import generate_password_hash


# ============================================================
# FLASK-LOGIN : chargement de l'utilisateur
# ============================================================

@login_manager.user_loader
def load_user(user_id):
    """Charge un utilisateur par son ID (requis par Flask-Login)"""
    return db.session.get(User, int(user_id))


# ============================================================
# ROUTES — Phase 1
# ============================================================

@app.route('/')
def dashboard():
    """
    Page d'accueil — Tableau de bord.
    Affiche les statistiques de présence du jour.
    """
    now = datetime.now()
    today = now.date()

    # Compter les employés actifs
    total_employees = Employee.query.filter_by(is_active=True).count()

    # Compter les employés qui ont pointé aujourd'hui
    present_today = db.session.query(
        Attendance.employee_id
    ).filter(
        Attendance.date == today
    ).distinct().count()

    # Calculer les absents
    absents = total_employees - present_today

    # Les retards seront calculés dans une phase ultérieure
    retards = 0

    # Préparer les statistiques
    stats = {
        'total': total_employees,
        'presents': present_today,
        'absents': absents,
        'retards': retards
    }

    # Récupérer les 10 derniers pointages du jour
    derniers_pointages = db.session.query(
        Attendance, Employee
    ).join(
        Employee
    ).filter(
        Attendance.date == today
    ).order_by(
        Attendance.check_in.desc()
    ).limit(10).all()

    # Formater la date du jour
    import locale
    today_str = now.strftime('%d/%m/%Y %H:%M:%S')

    return render_template(
        'dashboard.html',
        stats=stats,
        derniers_pointages=derniers_pointages,
        today=today_str
    )


# ============================================================
# INITIALISATION DE LA BASE DE DONNÉES
# ============================================================

def init_db():
    """
    Crée les tables et le compte admin par défaut.
    Cette fonction s'exécute au démarrage de l'application.
    """
    db.create_all()

    # Vérifier si l'admin existe déjà
    admin = User.query.filter_by(username=Config.DEFAULT_ADMIN_USER).first()
    if not admin:
        # Créer l'admin par défaut
        admin = User(
            username=Config.DEFAULT_ADMIN_USER,
            password_hash=generate_password_hash(Config.DEFAULT_ADMIN_PASS),
            role='admin'
        )
        db.session.add(admin)
        db.session.commit()
        print(f'[OK] Compte admin cree : {Config.DEFAULT_ADMIN_USER} / {Config.DEFAULT_ADMIN_PASS}')
    else:
        print(f'[OK] Compte admin existant : {Config.DEFAULT_ADMIN_USER}')


# Initialiser la base de données au démarrage
with app.app_context():
    init_db()


# ============================================================
# DÉMARRAGE
# ============================================================

if __name__ == '__main__':
    print('')
    print('=' * 50)
    print('  SAMHA METAL - Systeme de Presence')
    print('  http://localhost:5000')
    print('=' * 50)
    print('')
    app.run(debug=True, host='0.0.0.0', port=5000)
