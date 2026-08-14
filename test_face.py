"""
Script de test — Reconnaissance Faciale

Ce script teste le moteur de reconnaissance faciale étape par étape :
1. Charge le moteur InsightFace
2. Ouvre la webcam
3. Détecte les visages
4. Affiche l'embedding (vecteur 512D)
5. Sauvegarde une image annotée

Utilisation :
    python test_face.py
"""
import cv2
import sys
from face_engine import FaceEngine


def main():
    # ---- Étape 1 : Charger le moteur ----
    print('')
    print('=' * 50)
    print('  TEST - Moteur de Reconnaissance Faciale')
    print('=' * 50)
    print('')
    print('[1/4] Chargement du moteur InsightFace...')
    print('       (premier lancement = telechargement du modele ~300 MB)')
    print('')

    engine = FaceEngine()
    print('')

    # ---- Étape 2 : Capturer une image depuis la webcam ----
    print('[2/4] Ouverture de la webcam...')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('[ERREUR] Impossible d\'ouvrir la webcam.')
        print('         Verifiez que la webcam est connectee.')
        sys.exit(1)

    # Laisser la webcam s'initialiser
    for _ in range(10):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print('[ERREUR] Impossible de capturer une image.')
        sys.exit(1)

    print(f'       Image capturee : {frame.shape[1]}x{frame.shape[0]} pixels')
    print('')

    # ---- Étape 3 : Détecter les visages ----
    print('[3/4] Detection des visages...')
    faces = engine.detect(frame)

    if not faces:
        print('       Aucun visage detecte.')
        print('       Assurez-vous d\'etre face a la webcam.')
        # Sauvegarder l'image brute pour vérifier
        cv2.imwrite('data/test_no_face.jpg', frame)
        print('       Image sauvegardee : data/test_no_face.jpg')
        sys.exit(1)

    print(f'       {len(faces)} visage(s) detecte(s)')
    for i, face in enumerate(faces):
        box = face.bbox.astype(int)
        print(f'       Visage {i+1}:')
        print(f'         - Position : ({box[0]}, {box[1]}) -> ({box[2]}, {box[3]})')
        print(f'         - Score de detection : {face.det_score:.4f}')
    print('')

    # ---- Étape 4 : Extraire l'embedding ----
    print('[4/4] Extraction de l\'embedding ArcFace...')
    embedding = engine.get_embedding(frame)

    if embedding is not None:
        print(f'       Taille du vecteur : {embedding.shape[0]} dimensions')
        print(f'       Norme du vecteur  : {float(embedding @ embedding)**0.5:.4f}')
        print(f'       Premiers 10 valeurs : {embedding[:10].round(4).tolist()}')
    print('')

    # ---- Sauvegarder l'image annotée ----
    annotated = engine.draw_faces(frame, faces)
    output_path = 'data/test_detection.jpg'
    cv2.imwrite(output_path, annotated)
    print(f'[OK] Image annotee sauvegardee : {output_path}')
    print('')

    # ---- Test de comparaison ----
    print('[BONUS] Test de comparaison (meme visage vs lui-meme)...')
    score = engine.compare(embedding, embedding)
    print(f'        Similarite avec soi-meme : {score:.4f} (attendu: 1.0)')
    print('')

    print('=' * 50)
    print('  TEST TERMINE AVEC SUCCES')
    print('=' * 50)


if __name__ == '__main__':
    main()
