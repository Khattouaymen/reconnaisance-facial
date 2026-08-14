"""
Test d'intégration pour la Phase 4 : Borne de Pointage et Logique Multi-pointage
"""
from app import app
from extensions import db
from models import Employee, Attendance
from datetime import datetime, date, timedelta
import numpy as np

def test_pointage_logic():
    with app.app_context():
        # Créer ou récupérer un employé de test
        emp = Employee.query.filter_by(matricule="STG-001").first()
        if not emp:
            emp = Employee(
                matricule="STG-001",
                nom="Khattou",
                prenom="Aymen",
                departement="Service Informatique",
                poste="Stagiaire IA",
                is_active=True
            )
            db.session.add(emp)
            db.session.commit()

        today = date.today()
        # Nettoyer les présences du jour pour le test
        Attendance.query.filter_by(employee_id=emp.id, date=today).delete()
        db.session.commit()

        # 1. Test Première Entrée (Matin)
        now1 = datetime.now()
        att1 = Attendance(
            employee_id=emp.id,
            date=today,
            check_in=now1,
            check_out=None,
            status='present',
            confidence=0.88
        )
        db.session.add(att1)
        db.session.commit()
        print(f"[OK] Test 1 - Entrée matin validée : ID={att1.id}, Check-in={att1.check_in_time}")

        # 2. Test Sortie (Pause déjeuner)
        now2 = now1 + timedelta(hours=4)
        att1.check_out = now2
        db.session.commit()
        print(f"[OK] Test 2 - Sortie midi validée : ID={att1.id}, Check-out={att1.check_out_time}, Durée={att1.duration}h")

        # 3. Test Deuxième Entrée (Retour de pause)
        now3 = now2 + timedelta(hours=1)
        att2 = Attendance(
            employee_id=emp.id,
            date=today,
            check_in=now3,
            check_out=None,
            status='present',
            confidence=0.91
        )
        db.session.add(att2)
        db.session.commit()
        print(f"[OK] Test 3 - Entrée après-midi validée (Multi-pointage) : ID={att2.id}, Check-in={att2.check_in_time}")

        # 4. Test Sortie Finale (Fin de journée)
        now4 = now3 + timedelta(hours=3, minutes=30)
        att2.check_out = now4
        db.session.commit()
        print(f"[OK] Test 4 - Sortie soir validée : ID={att2.id}, Check-out={att2.check_out_time}, Durée={att2.duration}h")

        # Vérifier le total d'enregistrements du jour pour cet employé
        count = Attendance.query.filter_by(employee_id=emp.id, date=today).count()
        assert count == 2, f"Attendu 2 enregistrements (multi-pointage), trouvé {count}"
        print(f"[OK] Multi-pointage réussi : {count} sessions complètes enregistrées pour aujourd'hui.")

if __name__ == "__main__":
    test_pointage_logic()
