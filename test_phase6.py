"""
Test de validation de la Phase 6 : Authentification et Sécurisation Flask-Login
"""
from app import app
from extensions import db
from models import User

def test_auth_security():
    client = app.test_client()

    # 1. Test : Accès à une page protégée sans être connecté
    res_dash = client.get('/', follow_redirects=False)
    print(f"[TEST 1] Accès à '/' non-authentifié -> Statut: {res_dash.status_code} (Attendu: 302 vers /login)")
    assert res_dash.status_code == 302
    assert '/login' in res_dash.headers['Location']

    # 2. Test : Accès à la borne de pointage publique (sans login)
    res_pointage = client.get('/pointage')
    print(f"[TEST 2] Accès public à '/pointage' -> Statut: {res_pointage.status_code} (Attendu: 200)")
    assert res_pointage.status_code == 200

    # 3. Test : Connexion avec mauvais mot de passe
    res_bad_login = client.post('/login', data={'username': 'admin', 'password': 'wrong_password'}, follow_redirects=True)
    print(f"[TEST 3] Connexion erronée -> Statut: {res_bad_login.status_code}")
    assert b"Identifiant ou mot de passe incorrect" in res_bad_login.data

    # 4. Test : Connexion réussie avec admin / samha2025
    res_good_login = client.post('/login', data={'username': 'admin', 'password': 'samha2025'}, follow_redirects=True)
    print(f"[TEST 4] Connexion réussie -> Statut: {res_good_login.status_code}")
    assert b"SAMHA METAL" in res_good_login.data
    assert b"Tableau de bord" in res_good_login.data

    # 5. Test : Accès aux employés et rapports avec session active
    res_emp = client.get('/employees')
    print(f"[TEST 5] Accès à '/employees' après login -> Statut: {res_emp.status_code} (Attendu: 200)")
    assert res_emp.status_code == 200

    # 6. Test : Déconnexion
    res_logout = client.get('/logout', follow_redirects=False)
    print(f"[TEST 6] Déconnexion -> Statut: {res_logout.status_code} (Attendu: 302 vers /login)")
    assert res_logout.status_code == 302

    print("\n[OK] Tout le système d'authentification et de protection des routes est validé !")

if __name__ == "__main__":
    test_auth_security()
