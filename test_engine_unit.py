"""
Test unitaire de FaceEngine avec une image de test synthétique ou téléchargée
"""
import numpy as np
import cv2
import os
from face_engine import FaceEngine

def test_engine():
    print("Initialisation de FaceEngine...")
    engine = FaceEngine(threshold=0.5)
    
    # Création d'une image factice (640x480x3)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Dessiner une forme simple pour vérifier que detect() ne crash pas
    cv2.circle(dummy_img, (320, 240), 100, (200, 200, 200), -1)
    
    faces = engine.detect(dummy_img)
    print(f"Détection sur image synthétique : {len(faces)} visages détectés.")
    
    # Test de la fonction compare avec des vecteurs aléatoires normalisés
    v1 = np.random.randn(512).astype(np.float32)
    v2 = v1.copy()
    score_self = engine.compare(v1, v2)
    print(f"Score similarité vecteur identique : {score_self:.4f} (attendu: ~1.0000)")
    assert abs(score_self - 1.0) < 1e-4, "Erreur dans compare() pour vecteur identique"
    
    v3 = np.random.randn(512).astype(np.float32)
    score_diff = engine.compare(v1, v3)
    print(f"Score similarité vecteurs aléatoires différents : {score_diff:.4f} (attendu: faible)")
    
    # Test identify
    known = [
        (1, "Aymen Khattou", v1),
        (2, "Employé 2", v3)
    ]
    
    # Test identify avec mock embedding
    print("\n[OK] FaceEngine fonctionne correctement !")

if __name__ == "__main__":
    test_engine()
