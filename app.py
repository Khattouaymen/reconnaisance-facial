"""
Application Flask — Système de Présence SAMHA METAL S.A.R.L

Gère :
- Le tableau de bord (Phase 1)
- Le moteur de reconnaissance faciale InsightFace (Phase 2)
- La gestion des employés CRUD & Empreinte IA (Phase 3)
- La borne de pointage en temps réel avec multi-pointage intelligent (Phase 4)
"""
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from config import Config
from extensions import db, login_manager
from models import Employee, Attendance, User
from face_engine import FaceEngine
from werkzeug.security import generate_password_hash
from datetime import date, datetime
import os
import io
import base64
import numpy as np
import cv2


# ============================================================
# INITIALISATION DE L'APPLICATION
# ============================================================

app = Flask(__name__)
app.config.from_object(Config)

# Connecter les extensions
db.init_app(app)
login_manager.init_app(app)

# Dossier pour les photos des employés
FACES_DIR = app.config.get('UPLOAD_FOLDER', os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'faces'))
os.makedirs(FACES_DIR, exist_ok=True)

# Liste officielle des départements de SAMHA METAL
DEPARTEMENTS = [
    "Bureau d'Étude",
    "Atelier de Fabrication",
    "Équipe d'Intervention",
    "Administration et Finances",
    "Service Commercial",
    "Service Informatique",
    "Direction Générale"
]

# Initialisation du moteur IA InsightFace (SCRFD + ArcFace)
print('[IA] Chargement de FaceEngine...')
face_engine = FaceEngine(threshold=Config.FACE_SIMILARITY_THRESHOLD)


# ============================================================
# FLASK-LOGIN
# ============================================================

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ============================================================
# ROUTE FICHIERS UPLOADÉS (Photos employés)
# ============================================================

@app.route('/uploads/faces/<path:filename>')
def uploaded_face(filename):
    """Sert les photos de profil des employés depuis data/faces/"""
    return send_from_directory(FACES_DIR, filename)


# ============================================================
# DASHBOARD (PHASE 1)
# ============================================================

@app.route('/')
def dashboard():
    """Page d'accueil — Statistiques de présence du jour"""
    now = datetime.now()
    today = now.date()

    total_employees = Employee.query.filter_by(is_active=True).count()

    present_today = db.session.query(
        Attendance.employee_id
    ).filter(
        Attendance.date == today
    ).distinct().count()

    absents = max(0, total_employees - present_today)
    retards = Attendance.query.filter_by(date=today, status='retard').count()

    stats = {
        'total': total_employees,
        'presents': present_today,
        'absents': absents,
        'retards': retards
    }

    derniers_pointages = db.session.query(
        Attendance, Employee
    ).join(
        Employee
    ).filter(
        Attendance.date == today
    ).order_by(
        Attendance.check_in.desc()
    ).limit(10).all()

    today_str = now.strftime('%d/%m/%Y')

    return render_template(
        'dashboard.html',
        stats=stats,
        derniers_pointages=derniers_pointages,
        today=today_str
    )


# ============================================================
# GESTION DES EMPLOYÉS (PHASE 3)
# ============================================================

@app.route('/employees')
def employees_list():
    """Liste filtrable et recherchable des employés"""
    query = request.args.get('q', '').strip()
    selected_dept = request.args.get('dept', '').strip()

    emp_query = Employee.query

    if query:
        search_filter = f"%{query}%"
        emp_query = emp_query.filter(
            (Employee.nom.ilike(search_filter)) |
            (Employee.prenom.ilike(search_filter)) |
            (Employee.matricule.ilike(search_filter))
        )

    if selected_dept:
        emp_query = emp_query.filter_by(departement=selected_dept)

    employees = emp_query.order_by(Employee.nom.asc()).all()

    return render_template(
        'employees.html',
        employees=employees,
        departements=DEPARTEMENTS,
        query=query,
        selected_dept=selected_dept
    )


@app.route('/employees/add', methods=['GET', 'POST'])
def employee_add():
    """Formulaire d'ajout d'employé avec capture faciale IA"""
    if request.method == 'POST':
        matricule = request.form.get('matricule', '').strip().upper()
        nom = request.form.get('nom', '').strip()
        prenom = request.form.get('prenom', '').strip()
        departement = request.form.get('departement', '').strip()
        poste = request.form.get('poste', '').strip()
        webcam_image = request.form.get('webcam_image', '')
        photo_file = request.files.get('photo_file')

        # Validation de base
        if not matricule or not nom or not prenom or not departement or not poste:
            flash("Veuillez remplir tous les champs obligatoires (*).", "error")
            return render_template('add_employee.html', departements=DEPARTEMENTS)

        # Vérifier l'unicité du matricule
        if Employee.query.filter_by(matricule=matricule).first():
            flash(f"Le matricule '{matricule}' est déjà utilisé par un autre employé.", "error")
            return render_template('add_employee.html', departements=DEPARTEMENTS)

        # Traitement de l'image (webcam base64 ou fichier uploadé)
        img_np = None

        if webcam_image and ',' in webcam_image:
            try:
                header, encoded = webcam_image.split(',', 1)
                img_data = base64.b64decode(encoded)
                nparr = np.frombuffer(img_data, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"[ERREUR] Décodage image webcam : {e}")

        elif photo_file and photo_file.filename != '':
            try:
                file_bytes = np.frombuffer(photo_file.read(), np.uint8)
                img_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"[ERREUR] Décodage fichier photo : {e}")

        photo_filename = None
        face_embedding_bytes = None

        # Si une image est fournie, extraire le visage via InsightFace
        if img_np is not None:
            faces = face_engine.detect(img_np)
            if not faces:
                flash("Attention : Aucun visage n'a été détecté sur l'image. Veuillez utiliser une photo bien cadrée et nette.", "error")
                return render_template('add_employee.html', departements=DEPARTEMENTS)

            embedding = face_engine.get_embedding(img_np)
            if embedding is not None:
                face_embedding_bytes = embedding.astype(np.float32).tobytes()

            photo_filename = f"emp_{matricule.lower()}.jpg"
            save_path = os.path.join(FACES_DIR, photo_filename)
            cv2.imwrite(save_path, img_np)
        else:
            flash("Veuillez capturer une photo via la webcam ou importer un fichier image.", "warning")
            return render_template('add_employee.html', departements=DEPARTEMENTS)

        # Créer et enregistrer l'employé
        new_employee = Employee(
            matricule=matricule,
            nom=nom,
            prenom=prenom,
            departement=departement,
            poste=poste,
            photo_path=photo_filename,
            face_embedding=face_embedding_bytes,
            is_active=True
        )

        try:
            db.session.add(new_employee)
            db.session.commit()
            flash(f"L'employé {prenom} {nom} ({matricule}) a été enregistré avec son empreinte faciale IA !", "success")
            return redirect(url_for('employees_list'))
        except Exception as e:
            db.session.rollback()
            flash(f"Erreur : Le matricule '{matricule}' existe déjà ou une erreur de base de données est survenue.", "error")
            return render_template('add_employee.html', departements=DEPARTEMENTS)

    return render_template('add_employee.html', departements=DEPARTEMENTS)


@app.route('/employees/<int:id>')
def employee_detail(id):
    """Fiche détaillée d'un employé avec son historique individuel"""
    employee = db.session.get(Employee, id)
    if not employee:
        flash("Employé introuvable.", "error")
        return redirect(url_for('employees_list'))

    attendances = Attendance.query.filter_by(
        employee_id=employee.id
    ).order_by(
        Attendance.date.desc(), Attendance.check_in.desc()
    ).limit(30).all()

    return render_template('employee_detail.html', employee=employee, attendances=attendances)


@app.route('/employees/<int:id>/toggle', methods=['POST'])
def employee_toggle(id):
    """Active ou désactive un employé"""
    employee = db.session.get(Employee, id)
    if employee:
        employee.is_active = not employee.is_active
        db.session.commit()
        status_str = "activé" if employee.is_active else "désactivé"
        flash(f"L'employé {employee.nom_complet} a été {status_str}.", "info")
    return redirect(url_for('employee_detail', id=id))


@app.route('/employees/<int:id>/delete', methods=['POST'])
def employee_delete(id):
    """Supprime un employé et ses pointages associés"""
    employee = db.session.get(Employee, id)
    if employee:
        nom_complet = employee.nom_complet
        Attendance.query.filter_by(employee_id=employee.id).delete()
        if employee.photo_path:
            p = os.path.join(FACES_DIR, employee.photo_path)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        db.session.delete(employee)
        db.session.commit()
        flash(f"L'employé {nom_complet} a été supprimé.", "info")
    return redirect(url_for('employees_list'))


# ============================================================
# POINTAGE TEMPS RÉEL (PHASE 4)
# ============================================================

@app.route('/pointage')
def pointage_page():
    """Page de la borne de pointage avec webcam en direct"""
    today = date.today()
    today_attendances = db.session.query(
        Attendance
    ).filter(
        Attendance.date == today
    ).order_by(
        Attendance.check_in.desc()
    ).limit(20).all()

    return render_template('pointage.html', today_attendances=today_attendances)


@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    """
    Endpoint de reconnaissance faciale en temps réel.
    Reçoit une frame vidéo (base64), identifie l'employé et enregistre le pointage.
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'status': 'error', 'message': 'Aucune image fournie'}), 400

    img_data_str = data['image']
    if ',' in img_data_str:
        img_data_str = img_data_str.split(',', 1)[1]

    try:
        raw_bytes = base64.b64decode(img_data_str)
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Erreur décodage image: {e}'}), 400

    if img_np is None:
        return jsonify({'status': 'error', 'message': 'Image invalide'}), 400

    # 1. Récupérer tous les employés actifs avec une empreinte faciale
    active_employees = Employee.query.filter_by(is_active=True).filter(
        Employee.face_embedding.isnot(None)
    ).all()

    if not active_employees:
        return jsonify({
            'status': 'no_employees',
            'message': 'Aucun employé actif enregistré.'
        })

    # 2. Préparer la liste des embeddings connus
    known_embeddings = []
    for emp in active_employees:
        emb_arr = np.frombuffer(emp.face_embedding, dtype=np.float32)
        known_embeddings.append((emp.id, emp.nom_complet, emb_arr))

    # 3. Identifier le visage via InsightFace
    match = face_engine.identify(img_np, known_embeddings, threshold=Config.FACE_SIMILARITY_THRESHOLD)

    if not match:
        faces = face_engine.detect(img_np)
        if not faces:
            return jsonify({'status': 'no_face', 'message': 'Aucun visage détecté'})
        return jsonify({'status': 'no_match', 'message': 'Visage inconnu'})

    # 4. Employé reconnu !
    emp_id = match['id']
    employee = db.session.get(Employee, emp_id)
    if not employee:
        return jsonify({'status': 'error', 'message': 'Employé introuvable'}), 404

    now = datetime.now()
    today = now.date()

    # 5. Gestion du Cooldown (anti-doublon) & Multi-Pointage
    last_attendance = Attendance.query.filter_by(
        employee_id=employee.id,
        date=today
    ).order_by(
        Attendance.check_in.desc()
    ).first()

    # Cooldown : vérifier si le dernier pointage (entrée ou sortie) a eu lieu il y a moins de X secondes
    if last_attendance:
        last_time = last_attendance.check_out if last_attendance.check_out else last_attendance.check_in
        time_elapsed = (now - last_time).total_seconds()
        if time_elapsed < Config.POINTAGE_COOLDOWN:
            seconds_left = int(Config.POINTAGE_COOLDOWN - time_elapsed)
            return jsonify({
                'status': 'cooldown',
                'employee': employee.nom_complet,
                'seconds_left': seconds_left,
                'message': f"Pointage déjà enregistré il y a {int(time_elapsed)}s. Veuillez patienter {seconds_left}s."
            })

    # 6. Logique Multi-pointage (Entrée ou Sortie)
    if not last_attendance or last_attendance.check_out is not None:
        # NOUVELLE ENTRÉE (Matin ou Retour de pause)
        # Calcul du retard : uniquement sur la première entrée de la journée
        first_entry_today = Attendance.query.filter_by(employee_id=employee.id, date=today).count() == 0
        is_late = first_entry_today and (now.strftime('%H:%M') > Config.WORK_START_TIME)
        status_presence = 'retard' if is_late else 'present'

        new_att = Attendance(
            employee_id=employee.id,
            date=today,
            check_in=now,
            check_out=None,
            status=status_presence,
            confidence=match['score']
        )
        db.session.add(new_att)
        db.session.commit()

        action_type = "ENTREE"
        action_label = "Entrée enregistrée"
    else:
        # SORTIE (Pause déjeuner ou Fin de journée)
        last_attendance.check_out = now
        db.session.commit()

        action_type = "SORTIE"
        action_label = "Sortie enregistrée"
        status_presence = last_attendance.status

    photo_url = url_for('uploaded_face', filename=employee.photo_path) if employee.photo_path else None

    return jsonify({
        'status': 'success',
        'action': action_type,
        'action_label': action_label,
        'employee': {
            'id': employee.id,
            'matricule': employee.matricule,
            'nom': employee.nom_complet,
            'departement': employee.departement,
            'poste': employee.poste,
            'photo_url': photo_url
        },
        'time': now.strftime('%H:%M:%S'),
        'confidence': round(match['score'] * 100, 1),
        'status_presence': status_presence
    })


# ============================================================
# INITIALISATION DE LA BASE DE DONNÉES
# ============================================================

def init_db():
    """Crée les tables et le compte admin par défaut"""
    db.create_all()
    admin = User.query.filter_by(username=Config.DEFAULT_ADMIN_USER).first()
    if not admin:
        admin = User(
            username=Config.DEFAULT_ADMIN_USER,
            password_hash=generate_password_hash(Config.DEFAULT_ADMIN_PASS),
            role='admin'
        )
        db.session.add(admin)
        db.session.commit()
        print(f'[OK] Compte admin cree : {Config.DEFAULT_ADMIN_USER} / {Config.DEFAULT_ADMIN_PASS}')


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
