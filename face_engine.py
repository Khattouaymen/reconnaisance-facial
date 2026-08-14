"""
Moteur de Reconnaissance Faciale — SAMHA METAL S.A.R.L

Utilise InsightFace qui combine :
- SCRFD  : détection de visages (localise les visages dans l'image)
- ArcFace : reconnaissance faciale (génère un vecteur 512D par visage)

Le principe :
1. On détecte les visages dans une image (SCRFD)
2. On extrait un vecteur de 512 nombres pour chaque visage (ArcFace)
3. On compare ce vecteur avec les vecteurs des employés connus
4. Si la similarité dépasse le seuil → on identifie la personne
"""
import numpy as np
import cv2
from insightface.app import FaceAnalysis


class FaceEngine:
    """
    Classe principale pour la détection et la reconnaissance de visages.

    Utilisation :
        engine = FaceEngine()
        embedding = engine.get_embedding(image)
        result = engine.identify(image, known_employees)
    """

    def __init__(self, model_name='buffalo_l', threshold=0.5):
        """
        Initialise le moteur.

        Args:
            model_name: Nom du modèle InsightFace ('buffalo_l' = haute précision)
            threshold: Seuil de similarité pour l'identification (0 à 1)
        """
        print('[FaceEngine] Chargement du modele...')
        self.app = FaceAnalysis(
            name=model_name,
            providers=['CPUExecutionProvider']
        )
        # det_size = taille de l'image pour la détection (plus grand = plus précis mais plus lent)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.threshold = threshold
        print('[FaceEngine] Modele charge.')

    # ----------------------------------------------------------
    # DÉTECTION
    # ----------------------------------------------------------

    def detect(self, image):
        """
        Détecte tous les visages dans une image.

        Args:
            image: Image numpy array (BGR, format OpenCV)

        Returns:
            Liste de visages. Chaque visage est un objet avec :
              - .bbox       : [x1, y1, x2, y2] coordonnées du rectangle
              - .det_score  : score de confiance (0 à 1)
              - .embedding  : vecteur 512D (numpy array)
              - .kps        : 5 points clés du visage (yeux, nez, bouche)
        """
        if image is None:
            return []
        faces = self.app.get(image)
        return faces

    # ----------------------------------------------------------
    # EXTRACTION D'EMBEDDING
    # ----------------------------------------------------------

    def get_embedding(self, image):
        """
        Extrait le vecteur 512D du visage principal dans l'image.

        Args:
            image: Image numpy array (BGR)

        Returns:
            numpy array de 512 valeurs, ou None si aucun visage détecté
        """
        faces = self.detect(image)
        if not faces:
            return None

        # Prendre le visage avec le meilleur score de détection
        best_face = max(faces, key=lambda f: f.det_score)
        return best_face.embedding

    # ----------------------------------------------------------
    # COMPARAISON
    # ----------------------------------------------------------

    def compare(self, embedding1, embedding2):
        """
        Calcule la similarité entre deux visages.

        Utilise la similarité cosinus :
        - 1.0  = visages identiques
        - 0.5+ = probablement la même personne
        - 0.0  = visages très différents

        Args:
            embedding1, embedding2: vecteurs numpy 512D

        Returns:
            Score de similarité (float entre -1 et 1)
        """
        # Normaliser les vecteurs
        e1 = embedding1 / np.linalg.norm(embedding1)
        e2 = embedding2 / np.linalg.norm(embedding2)
        # Produit scalaire = cosine similarity quand les vecteurs sont normalisés
        return float(np.dot(e1, e2))

    # ----------------------------------------------------------
    # IDENTIFICATION
    # ----------------------------------------------------------

    def identify(self, image, known_employees, threshold=None):
        """
        Identifie une personne parmi les employés enregistrés.

        Args:
            image: Image numpy array (BGR)
            known_employees: Liste de tuples (id, nom_complet, embedding)
            threshold: Seuil minimum (défaut: self.threshold)

        Returns:
            dict {'id': int, 'nom': str, 'score': float} ou None
        """
        if threshold is None:
            threshold = self.threshold

        if not known_employees:
            return None

        # Extraire l'embedding du visage dans l'image
        embedding = self.get_embedding(image)
        if embedding is None:
            return None

        # Comparer avec chaque employé connu
        best_id = None
        best_nom = None
        best_score = -1

        for emp_id, emp_nom, emp_embedding in known_employees:
            score = self.compare(embedding, emp_embedding)
            if score > best_score:
                best_score = score
                best_id = emp_id
                best_nom = emp_nom

        # Vérifier si le meilleur score dépasse le seuil
        if best_score >= threshold:
            return {
                'id': best_id,
                'nom': best_nom,
                'score': round(best_score, 4)
            }

        return None

    # ----------------------------------------------------------
    # UTILITAIRES
    # ----------------------------------------------------------

    def draw_faces(self, image, faces):
        """
        Dessine les rectangles et scores sur les visages détectés.
        Utile pour le débogage.

        Args:
            image: Image numpy (BGR)
            faces: Liste de visages (retournée par detect())

        Returns:
            Image avec les annotations
        """
        img = image.copy()
        for face in faces:
            # Rectangle autour du visage
            box = face.bbox.astype(int)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 180, 0), 2)

            # Score de détection
            score_text = f'{face.det_score:.2f}'
            cv2.putText(img, score_text, (box[0], box[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1)

        return img
