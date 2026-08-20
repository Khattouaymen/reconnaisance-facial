"""
Application Flask — Système de Présence SAMHA METAL S.A.R.L

Gère :
- Le tableau de bord (Phase 1)
- Le moteur de reconnaissance faciale InsightFace (Phase 2)
- La gestion des employés CRUD & Empreinte IA (Phase 3)
- La borne de pointage en temps réel avec multi-pointage intelligent (Phase 4)
- L'historique avancé, les rapports d'assiduité & l'export CSV Excel (Phase 5)
- L'authentification et la sécurisation des accès administrateur (Phase 6)
"""
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, Response
from flask_login import login_user, logout_user, login_required, current_user
from config import Config
from extensions import db, login_manager
from models import Employee, Attendance, User
from face_engine import FaceEngine
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import date, datetime, timedelta
import os
import io
import csv
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
login_manager.login_view = 'login'
login_manager.login_message = "Veuillez vous connecter pour accéder à l'espace administration."
login_manager.login_message_category = "warning"

# Dossier pour les photos des employés
FACES_DIR = app.config.get('UPLOAD_FOLDER', os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'faces'))
os.makedirs(FACES_DIR, exist_ok=True)

# Liste officielle des 7 départements de SAMHA METAL
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
print('[IA] Initialisation du moteur FaceEngine...')
face_engine = FaceEngine(threshold=Config.FACE_SIMILARITY_THRESHOLD)


# ============================================================
# FONCTION UTILITAIRE : DÉCODAGE D'IMAGE
# ============================================================

def decode_image(data_or_file):
    """
    Décode une image provenant soit d'une chaîne base64 (webcam),
    soit d'un fichier uploadé (FileStorage).
    Retourne une image numpy array au format OpenCV BGR, ou None.
    """
    if not data_or_file:
        return None

    try:
        if isinstance(data_or_file, str) and ',' in data_or_file:
            _, encoded = data_or_file.split(',', 1)
            raw_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(raw_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if hasattr(data_or_file, 'read'):
            file_bytes = np.frombuffer(data_or_file.read(), np.uint8)
            return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERREUR] decode_image : {e}")

    return None


# ============================================================
# AUTHENTIFICATION & SESSIONS (PHASE 6)
# ============================================================

@login_manager.user_loader
def load_user(user_id):
    """Charge l'utilisateur en session par son identifiant unique"""
    return db.session.get(User, int(user_id))


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Page de connexion pour l'administrateur"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash(f"Bienvenue, {user.username} !", "success")
            next_page = request.args.get('next')
            return redirect(next_page if next_page else url_for('dashboard'))
        else:
            flash("Identifiant ou mot de passe incorrect.", "error")

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """Déconnexion de la session administrateur"""
    logout_user()
    flash("Vous avez été déconnecté avec succès.", "info")
    return redirect(url_for('login'))


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
@login_required
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
@login_required
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
@login_required
def employee_add():
    """Formulaire d'ajout d'employé avec capture faciale IA"""
    if request.method == 'POST':
        matricule = request.form.get('matricule', '').strip().upper()
        nom = request.form.get('nom', '').strip()
        prenom = request.form.get('prenom', '').strip()
        departement = request.form.get('departement', '').strip()
        poste = request.form.get('poste', '').strip()

        if not matricule or not nom or not prenom or not departement or not poste:
            flash("Veuillez remplir tous les champs obligatoires (*).", "error")
            return render_template('add_employee.html', departements=DEPARTEMENTS)

        if Employee.query.filter_by(matricule=matricule).first():
            flash(f"Le matricule '{matricule}' est déjà utilisé par un autre employé.", "error")
            return render_template('add_employee.html', departements=DEPARTEMENTS)

        raw_source = request.form.get('webcam_image') or request.files.get('photo_file')
        img_np = decode_image(raw_source)

        if img_np is None:
            flash("Veuillez capturer une photo via la webcam ou importer un fichier image valide.", "warning")
            return render_template('add_employee.html', departements=DEPARTEMENTS)

        faces = face_engine.detect(img_np)
        if not faces:
            flash("Attention : Aucun visage n'a été détecté sur l'image. Veuillez utiliser une photo bien cadrée et nette.", "error")
            return render_template('add_employee.html', departements=DEPARTEMENTS)

        embedding = face_engine.get_embedding(img_np)
        face_embedding_bytes = embedding.astype(np.float32).tobytes() if embedding is not None else None

        photo_filename = f"emp_{matricule.lower()}.jpg"
        save_path = os.path.join(FACES_DIR, photo_filename)
        cv2.imwrite(save_path, img_np)

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
        except Exception:
            db.session.rollback()
            flash(f"Erreur : Impossible d'enregistrer l'employé '{matricule}'.", "error")
            return render_template('add_employee.html', departements=DEPARTEMENTS)

    return render_template('add_employee.html', departements=DEPARTEMENTS)


@app.route('/employees/<int:id>')
@login_required
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
@login_required
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
@login_required
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
# BORNE DE POINTAGE TEMPS RÉEL (PHASE 4 - ACCÈS LIBRE)
# ============================================================

@app.route('/pointage')
def pointage_page():
    """Page de la borne de pointage avec webcam en direct (accessible pour les employés)"""
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

    img_np = decode_image(data['image'])
    if img_np is None:
        return jsonify({'status': 'error', 'message': 'Image invalide'}), 400

    # 1. Récupérer les employés actifs avec une empreinte faciale
    active_employees = Employee.query.filter_by(is_active=True).filter(
        Employee.face_embedding.isnot(None)
    ).all()

    if not active_employees:
        return jsonify({
            'status': 'no_employees',
            'message': 'Aucun employé actif enregistré.'
        })

    # 2. Préparer la liste des embeddings
    known_embeddings = []
    for emp in active_employees:
        emb_arr = np.frombuffer(emp.face_embedding, dtype=np.float32)
        known_embeddings.append((emp.id, emp.nom_complet, emb_arr))

    # 3. Identifier via InsightFace
    match = face_engine.identify(img_np, known_embeddings, threshold=Config.FACE_SIMILARITY_THRESHOLD)

    if not match:
        faces = face_engine.detect(img_np)
        if not faces:
            return jsonify({'status': 'no_face', 'message': 'Aucun visage détecté'})
        return jsonify({'status': 'no_match', 'message': 'Visage inconnu'})

    # 4. Employé reconnu
    emp_id = match['id']
    employee = db.session.get(Employee, emp_id)
    if not employee:
        return jsonify({'status': 'error', 'message': 'Employé introuvable'}), 404

    now = datetime.now()
    today = now.date()

    # 5. Cooldown 30s
    last_attendance = Attendance.query.filter_by(
        employee_id=employee.id,
        date=today
    ).order_by(
        Attendance.check_in.desc()
    ).first()

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

    # 6. Multi-pointage (Entrée ou Sortie)
    if not last_attendance or last_attendance.check_out is not None:
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
# HISTORIQUE & RAPPORTS (PHASE 5)
# ============================================================

@app.route('/history')
@login_required
def history_page():
    """Page d'historique global filtrable"""
    start_date = request.args.get('start_date', '').strip()
    end_date = request.args.get('end_date', '').strip()
    selected_dept = request.args.get('dept', '').strip()
    selected_emp_id = request.args.get('emp_id', type=int)
    selected_status = request.args.get('status', '').strip()

    query = Attendance.query.join(Employee)

    if start_date:
        try:
            d_start = datetime.strptime(start_date, '%Y-%m-%d').date()
            query = query.filter(Attendance.date >= d_start)
        except ValueError:
            pass

    if end_date:
        try:
            d_end = datetime.strptime(end_date, '%Y-%m-%d').date()
            query = query.filter(Attendance.date <= d_end)
        except ValueError:
            pass

    if selected_dept:
        query = query.filter(Employee.departement == selected_dept)

    if selected_emp_id:
        query = query.filter(Attendance.employee_id == selected_emp_id)

    if selected_status:
        query = query.filter(Attendance.status == selected_status)

    attendances = query.order_by(Attendance.date.desc(), Attendance.check_in.desc()).all()

    total_hours = round(sum(att.duration for att in attendances if att.duration), 2)
    total_retards = sum(1 for att in attendances if att.status == 'retard')

    all_employees = Employee.query.filter_by(is_active=True).order_by(Employee.nom.asc()).all()

    return render_template(
        'history.html',
        attendances=attendances,
        departements=DEPARTEMENTS,
        all_employees=all_employees,
        start_date=start_date,
        end_date=end_date,
        selected_dept=selected_dept,
        selected_emp_id=selected_emp_id,
        selected_status=selected_status,
        total_hours=total_hours,
        total_retards=total_retards
    )


@app.route('/reports')
@login_required
def reports_page():
    """Page de synthèse et rapports d'assiduité par période"""
    period = request.args.get('period', 'month').strip()
    today = date.today()

    if period == 'last_30':
        d_start = today - timedelta(days=30)
        d_end = today
    elif period == 'all':
        d_start = date(2020, 1, 1)
        d_end = today
    else:
        d_start = date(today.year, today.month, 1)
        d_end = today

    attendances = Attendance.query.join(Employee).filter(
        Attendance.date >= d_start,
        Attendance.date <= d_end
    ).all()

    total_registered_employees = Employee.query.filter_by(is_active=True).count()

    emp_map = {}
    for emp in Employee.query.filter_by(is_active=True).order_by(Employee.nom.asc()).all():
        emp_map[emp.id] = {
            'id': emp.id,
            'matricule': emp.matricule,
            'nom_complet': emp.nom_complet,
            'departement': emp.departement,
            'days': set(),
            'total_hours': 0.0,
            'retards_count': 0
        }

    for att in attendances:
        if att.employee_id in emp_map:
            emp_map[att.employee_id]['days'].add(att.date)
            if att.duration:
                emp_map[att.employee_id]['total_hours'] += att.duration
            if att.status == 'retard':
                emp_map[att.employee_id]['retards_count'] += 1

    emp_summary = []
    grand_total_hours = 0.0
    total_retards = 0
    active_employees_count = 0

    for emp_data in emp_map.values():
        days_present = len(emp_data['days'])
        hours = round(emp_data['total_hours'], 2)
        grand_total_hours += hours
        total_retards += emp_data['retards_count']
        if days_present > 0:
            active_employees_count += 1

        emp_summary.append({
            'id': emp_data['id'],
            'matricule': emp_data['matricule'],
            'nom_complet': emp_data['nom_complet'],
            'departement': emp_data['departement'],
            'days_present': days_present,
            'total_hours': hours,
            'retards_count': emp_data['retards_count']
        })

    emp_summary.sort(key=lambda x: x['total_hours'], reverse=True)

    dept_summary = []
    for dept_name in DEPARTEMENTS:
        dept_emps = [e for e in emp_summary if e['departement'] == dept_name]
        dept_hours = round(sum(e['total_hours'] for e in dept_emps), 2)
        dept_retards = sum(e['retards_count'] for e in dept_emps)
        dept_summary.append({
            'name': dept_name,
            'emp_count': len(dept_emps),
            'hours': dept_hours,
            'retards': dept_retards
        })

    mois_fr = ["", "Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
    current_month_name = f"{mois_fr[today.month]} {today.year}"

    return render_template(
        'reports.html',
        selected_period=period,
        start_date_str=d_start.strftime('%d/%m/%Y'),
        end_date_str=d_end.strftime('%d/%m/%Y'),
        current_month_name=current_month_name,
        grand_total_hours=round(grand_total_hours, 2),
        total_retards=total_retards,
        active_employees_count=active_employees_count,
        total_registered_employees=total_registered_employees,
        emp_summary=emp_summary,
        dept_summary=dept_summary
    )


@app.route('/attendance/export')
@login_required
def export_csv():
    """Exporte les présences au format CSV Excel"""
    start_date = request.args.get('start_date', '').strip()
    end_date = request.args.get('end_date', '').strip()
    selected_dept = request.args.get('dept', '').strip()
    selected_emp_id = request.args.get('emp_id', type=int)
    selected_status = request.args.get('status', '').strip()
    period = request.args.get('period', '').strip()

    query = Attendance.query.join(Employee)

    if period:
        today = date.today()
        if period == 'last_30':
            query = query.filter(Attendance.date >= today - timedelta(days=30))
        elif period == 'month':
            query = query.filter(Attendance.date >= date(today.year, today.month, 1))

    if start_date:
        try:
            query = query.filter(Attendance.date >= datetime.strptime(start_date, '%Y-%m-%d').date())
        except ValueError:
            pass

    if end_date:
        try:
            query = query.filter(Attendance.date <= datetime.strptime(end_date, '%Y-%m-%d').date())
        except ValueError:
            pass

    if selected_dept:
        query = query.filter(Employee.departement == selected_dept)

    if selected_emp_id:
        query = query.filter(Attendance.employee_id == selected_emp_id)

    if selected_status:
        query = query.filter(Attendance.status == selected_status)

    attendances = query.order_by(Attendance.date.desc(), Attendance.check_in.desc()).all()

    output = io.StringIO()
    output.write('\ufeff')
    writer = csv.writer(output, delimiter=';', quoting=csv.QUOTE_MINIMAL)

    writer.writerow([
        'Date',
        'Matricule',
        'Nom',
        'Prénom',
        'Département',
        'Poste',
        'Heure Entrée',
        'Heure Sortie',
        'Durée (h)',
        'Score IA (%)',
        'Statut'
    ])

    for att in attendances:
        conf_str = f"{round(att.confidence * 100, 1)}%" if att.confidence else "-"
        duration_str = str(att.duration) if att.duration else "En cours"
        statut_str = "En Retard" if att.status == 'retard' else "Présent"

        writer.writerow([
            att.date.strftime('%d/%m/%Y'),
            att.employee.matricule,
            att.employee.nom,
            att.employee.prenom,
            att.employee.departement,
            att.employee.poste,
            att.check_in_time,
            att.check_out_time,
            duration_str,
            conf_str,
            statut_str
        ])

    csv_data = output.getvalue()
    filename = f"presences_samha_metal_{date.today().strftime('%Y%m%d')}.csv"

    return Response(
        csv_data,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


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
