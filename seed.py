"""
Script de Démonstration (Seed Data) — SAMHA METAL S.A.R.L

Génère un jeu de données réaliste pour tester l'ensemble du système :
- 7 employés répartis dans les 7 départements officiels de SAMHA METAL
- Historique de pointages (entrées/sorties) sur les 14 derniers jours ouvrés

Utilisation :
    python seed.py
"""
import random
from datetime import date, datetime, timedelta
import numpy as np
from app import app
from extensions import db
from models import Employee, Attendance

# 7 Employés types représentant l'organigramme de SAMHA METAL
SAMPLE_EMPLOYEES = [
    {
        "matricule": "SM-DIR-01",
        "nom": "El Fassi",
        "prenom": "Mohammed",
        "departement": "Direction Générale",
        "poste": "Directeur Général"
    },
    {
        "matricule": "SM-BE-02",
        "nom": "Alami",
        "prenom": "Youssef",
        "departement": "Bureau d'Étude",
        "poste": "Ingénieur en Conception"
    },
    {
        "matricule": "SM-FAB-03",
        "nom": "Tazi",
        "prenom": "Hassan",
        "departement": "Atelier de Fabrication",
        "poste": "Chef d'Atelier Chaudronnerie"
    },
    {
        "matricule": "SM-INT-04",
        "nom": "Bennani",
        "prenom": "Omar",
        "departement": "Équipe d'Intervention",
        "poste": "Chef d'Équipe Montage Chantier"
    },
    {
        "matricule": "SM-FIN-05",
        "nom": "Chraibi",
        "prenom": "Fatima",
        "departement": "Administration et Finances",
        "poste": "Responsable Financière"
    },
    {
        "matricule": "SM-COM-06",
        "nom": "Idrissi",
        "prenom": "Salma",
        "departement": "Service Commercial",
        "poste": "Chargée de Clientèle Industrielle"
    },
    {
        "matricule": "SM-IT-07",
        "nom": "Khattou",
        "prenom": "Aymen",
        "departement": "Service Informatique",
        "poste": "Stagiaire Développeur IA"
    }
]

def seed_database():
    with app.app_context():
        print("[SEED] Initialisation des données de test SAMHA METAL...")

        created_employees = []

        for emp_data in SAMPLE_EMPLOYEES:
            existing = Employee.query.filter_by(matricule=emp_data["matricule"]).first()
            if not existing:
                # Vecteur d'empreinte faciale factice normalisé (512 dimensions)
                mock_vector = np.random.randn(512).astype(np.float32)
                mock_vector = mock_vector / np.linalg.norm(mock_vector)

                emp = Employee(
                    matricule=emp_data["matricule"],
                    nom=emp_data["nom"],
                    prenom=emp_data["prenom"],
                    departement=emp_data["departement"],
                    poste=emp_data["poste"],
                    face_embedding=mock_vector.tobytes(),
                    is_active=True
                )
                db.session.add(emp)
                created_employees.append(emp)
            else:
                created_employees.append(existing)

        db.session.commit()
        print(f"[SEED] {len(created_employees)} employés configurés.")

        # Générer 14 jours d'historique de présences réaliste
        today = date.today()
        total_created_att = 0

        for day_offset in range(14, -1, -1):
            current_day = today - timedelta(days=day_offset)
            
            # Pas de pointage le dimanche (jour de repos)
            if current_day.weekday() == 6:
                continue

            for emp in created_employees:
                # 90% de taux de présence moyen
                if random.random() > 0.12:
                    # Vérifier si déjà présent pour cette date
                    exists = Attendance.query.filter_by(employee_id=emp.id, date=current_day).first()
                    if exists:
                        continue

                    # 15% de chance d'être en retard (> 08:00)
                    is_late = random.random() < 0.15
                    if is_late:
                        minute_in = random.randint(5, 35)
                        hour_in = 8
                        status = "retard"
                    else:
                        minute_in = random.randint(45, 58)
                        hour_in = 7
                        status = "present"

                    # Session Matin (07:50 - 12:00)
                    in_time_morning = datetime(current_day.year, current_day.month, current_day.day, hour_in, minute_in)
                    out_time_morning = datetime(current_day.year, current_day.month, current_day.day, 12, random.randint(0, 10))

                    att_morning = Attendance(
                        employee_id=emp.id,
                        date=current_day,
                        check_in=in_time_morning,
                        check_out=out_time_morning if day_offset > 0 else None,
                        status=status,
                        confidence=round(random.uniform(0.72, 0.96), 4)
                    )
                    db.session.add(att_morning)
                    total_created_att += 1

                    # Session Après-midi (13:00 - 17:00) pour les jours passés
                    if day_offset > 0:
                        in_time_afternoon = datetime(current_day.year, current_day.month, current_day.day, 13, random.randint(0, 5))
                        out_time_afternoon = datetime(current_day.year, current_day.month, current_day.day, 17, random.randint(0, 20))

                        att_afternoon = Attendance(
                            employee_id=emp.id,
                            date=current_day,
                            check_in=in_time_afternoon,
                            check_out=out_time_afternoon,
                            status="present",
                            confidence=round(random.uniform(0.75, 0.95), 4)
                        )
                        db.session.add(att_afternoon)
                        total_created_att += 1

        db.session.commit()
        print(f"[SEED] {total_created_att} pointages réalistes créés avec succès !")
        print("[SEED] Base de données prête pour la démonstration !")

if __name__ == "__main__":
    seed_database()
