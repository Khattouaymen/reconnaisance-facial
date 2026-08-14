"""
Modèles de base de données — SAMHA METAL S.A.R.L

3 modèles :
- Employee  : les employés de l'entreprise
- Attendance : les enregistrements de présence (multi-pointage)
- User      : les comptes admin
"""
from extensions import db
from flask_login import UserMixin
from datetime import datetime, date


# ============================================================
# MODÈLE EMPLOYÉ
# ============================================================

class Employee(db.Model):
    """
    Représente un employé de SAMHA METAL.
    Chaque employé a un visage enregistré (embedding 512D) pour la reconnaissance.
    """
    __tablename__ = 'employees'

    id = db.Column(db.Integer, primary_key=True)
    matricule = db.Column(db.String(20), unique=True, nullable=False)
    nom = db.Column(db.String(80), nullable=False)
    prenom = db.Column(db.String(80), nullable=False)
    departement = db.Column(db.String(100), nullable=False)
    poste = db.Column(db.String(100), nullable=False)
    photo_path = db.Column(db.String(256))              # Chemin vers la photo de profil
    face_embedding = db.Column(db.LargeBinary)           # Vecteur 512D sérialisé (numpy)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relation : un employé a plusieurs enregistrements de présence
    attendances = db.relationship('Attendance', backref='employee', lazy=True)

    def __repr__(self):
        return f'<Employee {self.prenom} {self.nom}>'

    @property
    def nom_complet(self):
        """Retourne le nom complet de l'employé"""
        return f'{self.prenom} {self.nom}'


# ============================================================
# MODÈLE PRÉSENCE (MULTI-POINTAGE)
# ============================================================

class Attendance(db.Model):
    """
    Enregistrement de présence.

    Multi-pointage : un employé peut avoir PLUSIEURS enregistrements par jour.
    Chaque enregistrement = une paire entrée/sortie.

    Exemple d'une journée :
      Record #1 : check_in=08:00, check_out=12:00  (matin)
      Record #2 : check_in=13:00, check_out=17:00  (après-midi)
    """
    __tablename__ = 'attendances'

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'), nullable=False)
    date = db.Column(db.Date, default=date.today, nullable=False)
    check_in = db.Column(db.DateTime, nullable=False)
    check_out = db.Column(db.DateTime, nullable=True)   # NULL = pas encore sorti
    status = db.Column(db.String(20), default='present') # 'present' ou 'retard'
    confidence = db.Column(db.Float)                     # Score de similarité (0 à 1)

    def __repr__(self):
        return f'<Attendance {self.employee_id} {self.date} {self.status}>'

    @property
    def duration(self):
        """Calcule la durée de présence en heures pour cet enregistrement"""
        if self.check_out:
            delta = self.check_out - self.check_in
            return round(delta.total_seconds() / 3600, 2)  # En heures, arrondi
        return None

    @property
    def check_in_time(self):
        """Retourne l'heure de check_in au format HH:MM"""
        return self.check_in.strftime('%H:%M') if self.check_in else '--:--'

    @property
    def check_out_time(self):
        """Retourne l'heure de check_out au format HH:MM"""
        return self.check_out.strftime('%H:%M') if self.check_out else '--:--'


# ============================================================
# MODÈLE UTILISATEUR (ADMIN)
# ============================================================

class User(UserMixin, db.Model):
    """
    Compte administrateur pour accéder au système.
    UserMixin fournit les méthodes nécessaires pour Flask-Login.
    """
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='admin')

    def __repr__(self):
        return f'<User {self.username}>'
