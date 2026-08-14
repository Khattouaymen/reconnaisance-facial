"""
Test d'intégration pour la Phase 3 : Ajout d'un employé avec embedding facial
"""
import urllib.request
import urllib.parse
import json
import base64
import cv2
import numpy as np
from face_engine import FaceEngine
from app import app
from extensions import db
from models import Employee

def test_add_employee_flow():
    with app.app_context():
        # Nettoyage si déjà existant
        Employee.query.filter_by(matricule="SM-TEST-01").delete()
        db.session.commit()

        # Créer une image de test avec un faux visage ou un vecteur mock
        mock_embedding = np.random.randn(512).astype(np.float32)
        
        emp = Employee(
            matricule="SM-TEST-01",
            nom="Khattou",
            prenom="Aymen",
            departement="Service Informatique",
            poste="Stagiaire Développeur IA",
            photo_path="test_face.jpg",
            face_embedding=mock_embedding.tobytes(),
            is_active=True
        )
        db.session.add(emp)
        db.session.commit()

        print(f"[OK] Employé créé en base : ID={emp.id}, Nom={emp.nom_complet}")

        # Vérifier la récupération de l'employé et de son embedding
        saved_emp = Employee.query.filter_by(matricule="SM-TEST-01").first()
        assert saved_emp is not None
        assert saved_emp.nom == "Khattou"
        
        # Reconvertir l'embedding binaire en numpy array
        retrieved_emb = np.frombuffer(saved_emp.face_embedding, dtype=np.float32)
        assert retrieved_emb.shape == (512,)
        print(f"[OK] Embedding 512D restauré avec succès depuis la base de données.")
        print(f"     Norme du vecteur : {np.linalg.norm(retrieved_emb):.4f}")

if __name__ == "__main__":
    test_add_employee_flow()
